"""Minimal from-scratch (nn.Module) reimplementation of Qwen2.5-VL-3B-Instruct — model definition.

Contains the architecture config, preprocessing helpers, vision encoder, language
model, the full ``MiniQwen25VL`` module, and weight loading. It replicates the model
without using ``Qwen2_5_VLForConditionalGeneration`` / ``Qwen2_5_VLProcessor``:
pretrained weights are loaded straight from the safetensors checkpoint.

The inference / generation entry point lives in ``inference.py``.
"""
import math
import itertools
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

MODEL_PATH = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")

# ---- architecture config (from config.json / preprocessor_config.json) ----
class V:  # vision
    depth = 32
    hidden_size = 1280
    intermediate_size = 3420
    num_heads = 16
    in_chans = 3
    out_hidden_size = 2048
    patch_size = 14
    temporal_patch_size = 2
    spatial_merge_size = 2
    window_size = 112
    fullatt = (7, 15, 23, 31)


class T:  # text / llm
    hidden_size = 2048
    num_layers = 36
    num_heads = 16
    num_kv_heads = 2
    intermediate_size = 11008
    vocab_size = 151936
    rms_eps = 1e-6
    rope_theta = 1_000_000.0
    mrope_section = (16, 24, 24)  # temporal, height, width channel split (head_dim=128)


IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMG_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
EOS = (151645, 151643)
HEAD_DIM = T.hidden_size // T.num_heads  # 128
MERGE = V.spatial_merge_size  # 2
SMU = MERGE * MERGE  # 4


# ============================ preprocessing helpers ============================
def smart_resize(h, w, factor=28, min_px=56 * 56, max_px=12845056):
    """Qwen smart_resize: keep aspect ratio, both sides divisible by `factor`."""
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_px:
        b = math.sqrt(h * w / max_px)
        h_bar = max(factor, math.floor(h / b / factor) * factor)
        w_bar = max(factor, math.floor(w / b / factor) * factor)
    elif h_bar * w_bar < min_px:
        b = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * b / factor) * factor
        w_bar = math.ceil(w * b / factor) * factor
    return h_bar, w_bar


def vision_position_ids(grid_thw, merge):
    """(total_tokens, 2) -> (row, col) position per token; tokens in the same
    2x2 merge block share a position."""
    dev = grid_thw.device
    out = []
    for t, h, w in grid_thw.tolist():
        hp = torch.arange(h, device=dev).unsqueeze(1).expand(-1, w)
        hp = hp.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        wp = torch.arange(w, device=dev).unsqueeze(0).expand(h, -1)
        wp = wp.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        out.append(torch.stack([hp, wp], -1).repeat(t, 1))
    return torch.cat(out, 0)


def cu_seqlens(grid_thw):
    cs = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    return F.pad(cs, (1, 0), value=0)


def window_index(grid_thw, merge, window_size, patch_size):
    """Reorder indices so tokens of each attention window are contiguous."""
    dev = grid_thw.device
    idx, cu = [], [0]
    wid = 0
    mws = window_size // merge // patch_size  # 4
    for t, h, w in grid_thw.tolist():
        lh, lw = h // merge, w // merge
        index = torch.arange(t * lh * lw, device=dev).reshape(t, lh, lw)
        ph = mws - lh % mws
        pw = mws - lw % mws
        nwh, nww = (lh + ph) // mws, (lw + pw) // mws
        ip = F.pad(index, (0, pw, 0, ph), "constant", -100)
        ip = ip.reshape(t, nwh, mws, nww, mws).permute(0, 1, 3, 2, 4).reshape(t, nwh * nww, mws, mws)
        seqlens = (ip != -100).sum([2, 3]).reshape(-1)
        inew = ip.reshape(-1)
        inew = inew[inew != -100]
        idx.append(inew + wid)
        cu.extend((seqlens.cumsum(0) * SMU + cu[-1]).tolist())
        wid += t * lh * lw
    return torch.cat(idx, 0), torch.unique_consecutive(torch.tensor(cu, dtype=torch.int32, device=dev))


# ============================ shared modules ============================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(d)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


# ============================ vision encoder ============================
class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        k = (V.temporal_patch_size, V.patch_size, V.patch_size)
        self.proj = nn.Conv3d(V.in_chans, V.hidden_size, k, stride=k, bias=False)

    def forward(self, x):
        x = x.view(-1, V.in_chans, V.temporal_patch_size, V.patch_size, V.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, V.hidden_size)


class VisionRotary(nn.Module):
    def __init__(self, dim):  # dim = head_dim // 2 = 40
        super().__init__()
        self.dim = dim

    def forward(self, pos):  # pos: (L, 2)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.dim, 2, device=pos.device, dtype=torch.float) / self.dim))
        return (pos.unsqueeze(-1) * inv_freq).flatten(1)  # (L, 40)


class VisionAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = V.num_heads
        self.hd = V.hidden_size // V.num_heads
        self.qkv = nn.Linear(V.hidden_size, V.hidden_size * 3, bias=True)
        self.proj = nn.Linear(V.hidden_size, V.hidden_size)
        self.scale = self.hd ** -0.5

    def forward(self, x, cu, cos, sin):
        L = x.shape[0]
        q, k, v = self.qkv(x).reshape(L, 3, self.nh, self.hd).permute(1, 0, 2, 3).unbind(0)
        q, k = self._rope(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)  # (1, nh, L, hd)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        lens = (cu[1:] - cu[:-1]).tolist()
        qs, ks, vs = torch.split(q, lens, 2), torch.split(k, lens, 2), torch.split(v, lens, 2)
        out = torch.cat(
            [F.scaled_dot_product_attention(qi, ki, vi, scale=self.scale) for qi, ki, vi in zip(qs, ks, vs)],
            dim=2,
        )
        return self.proj(out.transpose(1, 2).reshape(L, -1))

    @staticmethod
    def _rope(q, k, cos, sin):
        od = q.dtype
        q, k = q.float(), k.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q.to(od), k.to(od)


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(V.hidden_size, V.intermediate_size, bias=True)
        self.up_proj = nn.Linear(V.hidden_size, V.intermediate_size, bias=True)
        self.down_proj = nn.Linear(V.intermediate_size, V.hidden_size, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(V.hidden_size)
        self.norm2 = RMSNorm(V.hidden_size)
        self.attn = VisionAttn()
        self.mlp = VisionMLP()

    def forward(self, x, cu, cos, sin):
        x = x + self.attn(self.norm1(x), cu, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Merger(nn.Module):
    def __init__(self):
        super().__init__()
        self.hs = V.hidden_size * SMU  # 1280 * 4
        self.ln_q = RMSNorm(V.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(self.hs, self.hs), nn.GELU(), nn.Linear(self.hs, V.out_hidden_size))

    def forward(self, x):
        return self.mlp(self.ln_q(x).view(-1, self.hs))


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.rotary_pos_emb = VisionRotary((V.hidden_size // V.num_heads) // 2)
        self.blocks = nn.ModuleList([VisionBlock() for _ in range(V.depth)])
        self.merger = Merger()

    def forward(self, x, grid_thw):
        pos = vision_position_ids(grid_thw, MERGE)
        cu = cu_seqlens(grid_thw)
        win, cuw = window_index(grid_thw, MERGE, V.window_size, V.patch_size)
        x = self.patch_embed(x)  # (L, hidden)
        L = x.shape[0]
        x = x.reshape(L // SMU, SMU, -1)[win].reshape(L, -1)
        rot = self.rotary_pos_emb(pos).reshape(L // SMU, SMU, -1)[win].reshape(L, -1)
        emb = torch.cat([rot, rot], -1)
        cos, sin = emb.cos(), emb.sin()
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu if i in V.fullatt else cuw, cos, sin)
        rev = torch.argsort(win)
        return self.merger(x)[rev]  # (num_merged, out_hidden)


# ============================ language model ============================
def apply_mrope(q, k, cos, sin):  # cos, sin: (b, 1, s, hd)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LLMAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = T.num_heads
        self.nkv = T.num_kv_heads
        self.ng = self.nh // self.nkv
        self.scale = HEAD_DIM ** -0.5
        self.q_proj = nn.Linear(T.hidden_size, self.nh * HEAD_DIM, bias=True)
        self.k_proj = nn.Linear(T.hidden_size, self.nkv * HEAD_DIM, bias=True)
        self.v_proj = nn.Linear(T.hidden_size, self.nkv * HEAD_DIM, bias=True)
        self.o_proj = nn.Linear(self.nh * HEAD_DIM, T.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache):
        b, q_len, _ = x.shape
        q = self.q_proj(x).view(b, q_len, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(b, q_len, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(b, q_len, self.nkv, HEAD_DIM).transpose(1, 2)
        q, k = apply_mrope(q, k, cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], 2)
            v = torch.cat([cache[1], v], 2)
        new = (k, v)
        k = repeat_kv(k, self.ng)
        v = repeat_kv(v, self.ng)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(q_len > 1), scale=self.scale)
        return self.o_proj(a.transpose(1, 2).reshape(b, q_len, -1)), new


class LLMMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(T.hidden_size, T.intermediate_size, bias=False)
        self.up_proj = nn.Linear(T.hidden_size, T.intermediate_size, bias=False)
        self.down_proj = nn.Linear(T.intermediate_size, T.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(T.hidden_size, T.rms_eps)
        self.post_attention_layernorm = RMSNorm(T.hidden_size, T.rms_eps)
        self.self_attn = LLMAttn()
        self.mlp = LLMMLP()

    def forward(self, x, cos, sin, cache):
        a, new = self.self_attn(self.input_layernorm(x), cos, sin, cache)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(T.vocab_size, T.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(T.num_layers)])
        self.norm = RMSNorm(T.hidden_size, T.rms_eps)

    def forward(self, x, cos, sin, cache=None):
        new_cache = []
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache else None
            x, nc = layer(x, cos, sin, c)
            new_cache.append(nc)
        return self.norm(x), new_cache


# ============================ full model ============================
class MiniQwen25VL(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer()
        self.model = TextModel()
        self.lm_head = nn.Linear(T.hidden_size, T.vocab_size, bias=False)
        self.sec = [s * 2 for s in T.mrope_section]  # [32, 48, 48]

    def rope_cos_sin(self, pos):  # pos: (3, b, s) long
        inv_freq = 1.0 / (T.rope_theta ** (torch.arange(0, HEAD_DIM, 2, device=pos.device, dtype=torch.float) / HEAD_DIM))
        freqs = pos.float().unsqueeze(-1) * inv_freq  # (3, b, s, 64)
        emb = torch.cat([freqs, freqs], -1)  # (3, b, s, 128)
        cos, sin = emb.cos(), emb.sin()
        cs, ss = cos.split(self.sec, -1), sin.split(self.sec, -1)
        cos = torch.cat([cs[i][i] for i in range(3)], -1).unsqueeze(1)  # (b, 1, s, 128)
        sin = torch.cat([ss[i][i] for i in range(3)], -1).unsqueeze(1)
        dt = self.lm_head.weight.dtype
        return cos.to(dt), sin.to(dt)

    def forward(self, inputs_embeds, pos, cache=None):
        cos, sin = self.rope_cos_sin(pos)
        h, new_cache = self.model(inputs_embeds, cos, sin, cache)
        return self.lm_head(h), new_cache

    @torch.no_grad()
    def generate(self, input_ids, mm_type, grid_thw, pixel_values, max_new=128, rep=1.05):
        dev = input_ids.device
        ie = self.visual(pixel_values, grid_thw)  # (n_img, hidden)
        embeds = self.model.embed_tokens(input_ids)
        mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1)
        embeds = embeds.masked_scatter(mask, ie.reshape(-1).to(embeds.dtype))
        pos, delta = rope_index(input_ids, mm_type, grid_thw)  # (3,1,L), int
        logits, cache = self.forward(embeds, pos)
        out = []
        cur = input_ids.shape[1]
        for _ in range(max_new):
            lg = logits[:, -1, :].float()
            all_ids = torch.cat([input_ids[0], torch.tensor(out, device=dev, dtype=torch.long)]) if out else input_ids[0]
            sc = lg[0, all_ids]
            lg[0, all_ids] = torch.where(sc < 0, sc * rep, sc / rep)
            nxt = lg.argmax(-1).item()
            if nxt in EOS:
                break
            out.append(nxt)
            cur += 1
            npos = torch.full((3, 1, 1), cur + delta, dtype=torch.long, device=dev)
            nem = self.model.embed_tokens(torch.tensor([[nxt]], device=dev))
            logits, cache = self.forward(nem, npos, cache)
        return out


def rope_index(input_ids, mm_type, grid_thw):
    """3D mrope position ids for a single (batch=1, no padding) sequence."""
    dev = input_ids.device
    ids = input_ids[0]
    L = ids.shape[0]
    mt = mm_type[0].tolist()
    groups = []
    for k, g in itertools.groupby(enumerate(mt), lambda x: x[1]):
        g = list(g)
        groups.append((k, g[0][0], g[-1][0] + 1))
    cur = 0
    parts = []
    gi = iter(grid_thw.tolist())
    for mod, s, e in groups:
        if mod == 0:  # text -> 1D positions
            n = e - s
            parts.append(torch.arange(n, device=dev).view(1, -1).expand(3, -1) + cur)
            cur += n
        else:  # image -> 3D positions (time_interval=1 for images)
            t, h, w = next(gi)
            lt, lh, lw = t, h // MERGE, w // MERGE
            pt = torch.arange(lt, device=dev).repeat_interleave(lh * lw) + cur
            ph = (torch.arange(lh, device=dev) + cur).repeat_interleave(lw).repeat(lt)
            pw = (torch.arange(lw, device=dev) + cur).repeat(lh * lt)
            parts.append(torch.stack([pt, ph, pw], 0))
            cur += max(h, w) // MERGE
    positions = torch.cat(parts, 1)  # (3, L)
    pos = torch.zeros(3, 1, L, dtype=torch.long, device=dev)
    pos[:, 0] = positions
    return pos, (positions.max() + 1 - L).item()


# ============================ weights ============================
def load_weights(model):
    sd = {}
    for f in sorted(glob.glob(MODEL_PATH + "/model-*.safetensors")):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.lm_head.weight = model.model.embed_tokens.weight  # tied embeddings
    return missing, unexpected
