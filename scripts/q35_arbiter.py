# Layer-0 linear_attn of the real Qwen3.8-27B: HF reference math (torch, pasted from
# modeling_qwen3_5.py) vs our transcription (numpy). First diverging stage = the bug.
import json, torch, numpy as np
import torch.nn.functional as F

def load(name, shape):
    b=open('t/linear_attn.'+name,'rb').read() if not name.startswith('input') else open('t/'+name,'rb').read()
    t=torch.frombuffer(bytearray(b), dtype=torch.bfloat16).reshape(*shape).float()
    return t

H=5120; NK,NV,DK,DV,CK = 16,48,128,128,4
KEY=NK*DK; VAL=NV*DV; CD=2*KEY+VAL
ids=json.load(open('t/ids.json'))
S=len(ids)
x = torch.frombuffer(bytearray(open('t/embed_rows.bin','rb').read()), dtype=torch.bfloat16).reshape(S,H).float()
ln1=load('input_layernorm.weight',[H])
Wqkv=load('linear_attn_in_proj_qkv.weight',[CD,H]) if False else load('in_proj_qkv.weight',[CD,H])
Wz=load('in_proj_z.weight',[VAL,H]); Wb=load('in_proj_b.weight',[NV,H]); Wa=load('in_proj_a.weight',[NV,H])
convw=load('conv1d.weight',[CD,1,CK]); dt=load('dt_bias',[NV]); Alog=load('A_log',[NV]); normw=load('norm.weight',[128])
Wout=load('out_proj.weight',[H,VAL])

# ---------------- torch reference (verbatim HF math) ----------------
def hf_rms(x,w,eps=1e-6):
    v=x.pow(2).mean(-1,keepdim=True)
    return x*torch.rsqrt(v+eps)*w
def l2norm(x,eps=1e-6):
    return x*torch.rsqrt((x*x).sum(-1,keepdim=True)+eps)

hn_t = hf_rms(x, ln1)
mixed = hn_t @ Wqkv.T                        # S, CD
z_t = (hn_t @ Wz.T)
b_t = hn_t @ Wb.T
a_t = hn_t @ Wa.T
# causal_conv1d_fn: conv1d(x[b,dim,seq], weight[dim,1,k], padding=k-1, groups=dim)[:, :, :seq] + silu
mx = mixed.T.unsqueeze(0)                    # 1, CD, S
conv_out = F.conv1d(mx, convw, padding=CK-1, groups=CD)[:, :, :S]
conv_t = F.silu(conv_out)[0].T               # S, CD
q = conv_t[:, :KEY].reshape(S, NK, DK)
k = conv_t[:, KEY:2*KEY].reshape(S, NK, DK)
v = conv_t[:, 2*KEY:].reshape(S, NV, DV)
beta = b_t.sigmoid()
g = -Alog.exp() * F.softplus(a_t + dt)
q = q.repeat_interleave(NV//NK, dim=1)
k = k.repeat_interleave(NV//NK, dim=1)
qn = l2norm(q); kn = l2norm(k)
qn = qn / (DK ** 0.5)
St = torch.zeros(NV, DK, DV)
core_t = torch.zeros(S, NV, DV)
for t in range(S):
    gt = g[t].exp()[:, None, None]
    St = St * gt
    kv_mem = (St * kn[t][:, :, None]).sum(-2)
    delta = (v[t] - kv_mem) * beta[t][:, None]
    St = St + kn[t][:, :, None] * delta[:, None, :]
    core_t[t] = (St * qn[t][:, :, None]).sum(-2)
gated_t = hf_rms(core_t, normw) * F.silu(z_t.reshape(S, NV, DV))
out_t = gated_t.reshape(S, VAL) @ Wout.T
res_t = x + out_t

# ---------------- our numpy transcription (same as engine) ----------------
def np_rms(x,w,eps=1e-6):
    return x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*w
def silu(x): return x/(1+np.exp(-x))
xn = np.array(x.tolist(), dtype=np.float32)
hn_n = np_rms(xn, np.array(ln1.tolist(),dtype=np.float32))
Wqkv_n=np.array(Wqkv.tolist(),dtype=np.float32)
mixed_n = hn_n @ Wqkv_n.T
cw = np.array(convw.tolist(),dtype=np.float32)[:,0,:]
padded = np.concatenate([np.zeros((CK-1,CD),np.float32), mixed_n],0)
conv_n = np.zeros_like(mixed_n)
for t in range(S):
    conv_n[t]=(padded[t:t+CK]*cw.T).sum(0)
conv_n = silu(conv_n)
qn_ = conv_n[:, :KEY].reshape(S,NK,DK); kn_ = conv_n[:, KEY:2*KEY].reshape(S,NK,DK); vn_ = conv_n[:, 2*KEY:].reshape(S,NV,DV)
def l2n(t): return t/np.sqrt((t*t).sum(-1,keepdims=True)+1e-6)
qn2=l2n(qn_); kn2=l2n(kn_)
qn2=np.repeat(qn2,NV//NK,axis=1)/np.sqrt(DK)
kn2=np.repeat(kn2,NV//NK,axis=1)
b_n = hn_n @ np.array(Wb.tolist(),dtype=np.float32).T
a_n = hn_n @ np.array(Wa.tolist(),dtype=np.float32).T
beta_n = 1/(1+np.exp(-b_n))
g_n = np.exp(-np.exp(np.array(Alog.tolist(),dtype=np.float32)) * np.log1p(np.exp(a_n+np.array(dt.tolist(),dtype=np.float32))))
Sn=np.zeros((NV,DK,DV),np.float32)
core_n=np.zeros((S,NV,DV),np.float32)
for t in range(S):
    Sn=Sn*g_n[t][:,None,None]
    kvm=(Sn*kn2[t][:,:,None]).sum(1)
    delta=(vn_[t]-kvm)*beta_n[t][:,None]
    Sn=Sn+kn2[t][:,:,None]*delta[:,None,:]
    core_n[t]=(Sn*qn2[t][:,:,None]).sum(1)
z_n = hn_n @ np.array(Wz.tolist(),dtype=np.float32).T
gated_n = np_rms(core_n, np.array(normw.tolist(),dtype=np.float32)) * silu(z_n.reshape(S,NV,DV))
out_n = gated_n.reshape(S,VAL) @ np.array(Wout.tolist(),dtype=np.float32).T
res_n = xn + out_n

def cmp(name, tt, nn):
    tt=np.array(tt.tolist(),dtype=np.float32)
    d=np.abs(tt-nn).max()
    print(f"{name:12s} maxdiff={d:.6f}  torch|.|={np.abs(tt).max():.4f}")
cmp("hn", hn_t, hn_n)
cmp("mixed", mixed, mixed_n)
cmp("conv", conv_t, conv_n)
cmp("beta", beta, beta_n)
cmp("g(exp)", g.exp(), g_n)
cmp("core", core_t, core_n)
cmp("gated", gated_t, gated_n)
cmp("out", out_t, out_n)
cmp("residual", res_t, res_n)
print("torch res last-token: |x|=%.3f first4=%s" % (res_t[-1].norm(), [round(float(v),4) for v in res_t[-1][:4]]))
print("HF ground truth      : |x|=13.494 first4=[0.041, -0.0315, 0.021, -0.0378]")
