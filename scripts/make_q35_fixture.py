# Builds a tiny random qwen3_5-style checkpoint + reference logits (NumPy transcription of the
# Hugging Face modeling_qwen3_5.py math) for validating the Rust implementation.
import json, struct, numpy as np

rng = np.random.default_rng(7)
import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "a"
if VARIANT == "a":
    H, I, L, NH, NKV, HD, V = 32, 64, 4, 4, 2, 16, 50
else:  # real-model ratios: 3 v-heads per k-head, gqa 3:1, bigger dims
    H, I, L, NH, NKV, HD, V = 64, 96, 8, 6, 2, 32, 64
ROT = HD // 4        # partial_rotary_factor 0.25
NK, NV, DK, DV, CK = (2, 4, 8, 8, 4) if VARIANT == "a" else (2, 6, 16, 16, 4)
KEY, VAL = NK*DK, NV*DV
CD = 2*KEY + VAL
THETA = 1e7
layer_types = (["linear_attention"]*3 + ["full_attention"]) * (L // 4)

def w(*shape, s=0.3):
    return (rng.standard_normal(shape) * s).astype(np.float32)

T = {}
pre = "model.language_model."
T[pre+"embed_tokens.weight"] = w(V, H)
T["lm_head.weight"] = w(V, H)
T[pre+"norm.weight"] = 1.0 + w(H, s=0.05)
for l in range(L):
    p = f"{pre}layers.{l}."
    T[p+"input_layernorm.weight"] = 1.0 + w(H, s=0.05)
    T[p+"post_attention_layernorm.weight"] = 1.0 + w(H, s=0.05)
    T[p+"mlp.gate_proj.weight"] = w(I, H)
    T[p+"mlp.up_proj.weight"] = w(I, H)
    T[p+"mlp.down_proj.weight"] = w(H, I)
    if layer_types[l] == "full_attention":
        T[p+"self_attn.q_proj.weight"] = w(NH*HD*2, H)
        T[p+"self_attn.k_proj.weight"] = w(NKV*HD, H)
        T[p+"self_attn.v_proj.weight"] = w(NKV*HD, H)
        T[p+"self_attn.o_proj.weight"] = w(H, NH*HD)
        T[p+"self_attn.q_norm.weight"] = 1.0 + w(HD, s=0.05)
        T[p+"self_attn.k_norm.weight"] = 1.0 + w(HD, s=0.05)
    else:
        T[p+"linear_attn.in_proj_qkv.weight"] = w(CD, H)
        T[p+"linear_attn.in_proj_z.weight"] = w(VAL, H)
        T[p+"linear_attn.in_proj_b.weight"] = w(NV, H)
        T[p+"linear_attn.in_proj_a.weight"] = w(NV, H)
        T[p+"linear_attn.conv1d.weight"] = w(CD, 1, CK)
        T[p+"linear_attn.dt_bias"] = np.abs(w(NV)) + 0.5
        T[p+"linear_attn.A_log"] = np.log(np.abs(w(NV)) + 0.5)
        T[p+"linear_attn.norm.weight"] = 1.0 + w(DV, s=0.05)
        T[p+"linear_attn.out_proj.weight"] = w(H, VAL)

def to_bf16(a):
    u = a.astype(np.float32).view(np.uint32)
    return (((u + 0x8000 - ((u >> 16) & 1 == 0)) if False else u) >> 16).astype(np.uint16)  # truncation, matches loader tolerance
def bf16_back(a):
    return (to_bf16(a).astype(np.uint32) << 16).view(np.float32)

# quantize weights to bf16 in BOTH the file and the reference so they match exactly
for k in T:
    T[k] = bf16_back(np.ascontiguousarray(T[k]))

import os
outdir = "/Users/pro/Desktop/llmfffff2/models/tiny-q35" + ("" if VARIANT == "a" else "-b")
os.makedirs(outdir, exist_ok=True)
# write safetensors
header = {}
off = 0
blobs = []
for name, a in T.items():
    b = to_bf16(a).tobytes()
    header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off+len(b)]}
    off += len(b)
    blobs.append(b)
hj = json.dumps(header).encode()
with open(outdir+"/model.safetensors","wb") as f:
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for b in blobs:
        f.write(b)
cfg = {"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",
 "text_config":{"model_type":"qwen3_5_text","hidden_size":H,"intermediate_size":I,"num_hidden_layers":L,
  "num_attention_heads":NH,"num_key_value_heads":NKV,"head_dim":HD,"vocab_size":V,"rms_norm_eps":1e-6,
  "attn_output_gate":True,"layer_types":layer_types,"partial_rotary_factor":0.25,
  "rope_parameters":{"rope_theta":THETA,"rope_type":"default","partial_rotary_factor":0.25},
  "linear_num_key_heads":NK,"linear_num_value_heads":NV,"linear_key_head_dim":DK,"linear_value_head_dim":DV,
  "linear_conv_kernel_dim":CK,"tie_word_embeddings":False}}
json.dump(cfg, open(outdir+"/config.json","w"))

# ---------------- NumPy reference forward ----------------
def rms(x, wgt, eps=1e-6):
    return x / np.sqrt((x*x).mean(-1, keepdims=True) + eps) * wgt
def silu(x): return x / (1 + np.exp(-x))
def softplus(x): return np.log1p(np.exp(x))

tokens = [3, 17, 42, 7, 5, 23, 11, 9, 31, 2, 44, 19]
S = len(tokens)
x = T[pre+"embed_tokens.weight"][tokens].astype(np.float32)   # S,H

inv = 1.0 / (THETA ** (np.arange(0, ROT, 2) / ROT))           # ROT/2
posf = np.arange(S)[:, None] * inv[None, :]                   # S, ROT/2
cos = np.cos(np.concatenate([posf, posf], -1))                # S, ROT
sin = np.sin(np.concatenate([posf, posf], -1))

for l in range(L):
    p = f"{pre}layers.{l}."
    hn = rms(x, T[p+"input_layernorm.weight"])
    if layer_types[l] == "full_attention":
        qg = hn @ T[p+"self_attn.q_proj.weight"].T            # S, NH*HD*2
        qg = qg.reshape(S, NH, 2*HD)
        q, gate = qg[:, :, :HD], qg[:, :, HD:]
        k = (hn @ T[p+"self_attn.k_proj.weight"].T).reshape(S, NKV, HD)
        v = (hn @ T[p+"self_attn.v_proj.weight"].T).reshape(S, NKV, HD)
        q = rms(q, T[p+"self_attn.q_norm.weight"])
        k = rms(k, T[p+"self_attn.k_norm.weight"])
        def rope(t):
            rot, pas = t[..., :ROT], t[..., ROT:]
            h1, h2 = rot[..., :ROT//2], rot[..., ROT//2:]
            rh = np.concatenate([-h2, h1], -1)
            return np.concatenate([rot*cos[:, None, :] + rh*sin[:, None, :], pas], -1)
        q, k = rope(q), rope(k)
        kk = np.repeat(k, NH//NKV, axis=1)                     # S, NH, HD
        vv = np.repeat(v, NH//NKV, axis=1)
        out = np.zeros((S, NH, HD), np.float32)
        for t in range(S):
            sc = (q[t, :, None, :] * kk[:t+1].transpose(1,0,2)).sum(-1) / np.sqrt(HD)  # NH, t+1
            sc = sc - sc.max(-1, keepdims=True)
            pr = np.exp(sc); pr /= pr.sum(-1, keepdims=True)
            out[t] = (pr[:, :, None] * vv[:t+1].transpose(1,0,2)).sum(1)
        out = out * (1/(1+np.exp(-gate)))
        x = x + out.reshape(S, NH*HD) @ T[p+"self_attn.o_proj.weight"].T
    else:
        qkv = hn @ T[p+"linear_attn.in_proj_qkv.weight"].T     # S, CD
        z = (hn @ T[p+"linear_attn.in_proj_z.weight"].T).reshape(S, NV, DV)
        b = hn @ T[p+"linear_attn.in_proj_b.weight"].T          # S, NV
        a = hn @ T[p+"linear_attn.in_proj_a.weight"].T
        # causal depthwise conv + silu
        cw = T[p+"linear_attn.conv1d.weight"][:, 0, :]          # CD, CK
        padded = np.concatenate([np.zeros((CK-1, CD), np.float32), qkv], 0)
        conv = np.zeros_like(qkv)
        for t in range(S):
            conv[t] = (padded[t:t+CK] * cw.T).sum(0)
        qkv = silu(conv)
        q = qkv[:, :KEY].reshape(S, NK, DK)
        k = qkv[:, KEY:2*KEY].reshape(S, NK, DK)
        v = qkv[:, 2*KEY:].reshape(S, NV, DV)
        def l2n(t): return t / np.sqrt((t*t).sum(-1, keepdims=True) + 1e-6)
        q, k = l2n(q), l2n(k)
        q = np.repeat(q, NV//NK, axis=1)                        # S, NV, DK
        k = np.repeat(k, NV//NK, axis=1)
        q = q / np.sqrt(DK)
        g = np.exp(-np.exp(T[p+"linear_attn.A_log"]) * softplus(a + T[p+"linear_attn.dt_bias"]))  # S,NV
        beta = 1/(1+np.exp(-b))
        st = np.zeros((NV, DK, DV), np.float32)
        core = np.zeros((S, NV, DV), np.float32)
        for t in range(S):
            st = st * g[t][:, None, None]
            kv_mem = (st * k[t][:, :, None]).sum(1)             # NV, DV
            delta = (v[t] - kv_mem) * beta[t][:, None]
            st = st + k[t][:, :, None] * delta[:, None, :]
            core[t] = (st * q[t][:, :, None]).sum(1)
        core = rms(core, T[p+"linear_attn.norm.weight"]) * silu(z)
        x = x + core.reshape(S, VAL) @ T[p+"linear_attn.out_proj.weight"].T
    hn = rms(x, T[p+"post_attention_layernorm.weight"])
    mlp = silu(hn @ T[p+"mlp.gate_proj.weight"].T) * (hn @ T[p+"mlp.up_proj.weight"].T)
    x = x + mlp @ T[p+"mlp.down_proj.weight"].T

xf = rms(x, T[pre+"norm.weight"])
logits = xf @ T["lm_head.weight"].T                             # S, V
with open(outdir+"/fixture.json","w") as f:
    json.dump({"tokens": tokens, "logits_last": [float(v) for v in logits[-1]], "argmax": [int(v) for v in logits.argmax(-1)]}, f)
print("fixture written:", outdir, "| last-token logits[:4]:", logits[-1][:4])
