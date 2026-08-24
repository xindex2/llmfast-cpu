# 1-layer checkpoint from the REAL Qwen3.8-27B layer-0 linear_attn, with real embedding rows
# remapped to token ids 0..58 and a zeroed MLP, so the Rust engine's layer-0 output can be
# compared against the torch reference at full dimensions.
import json, struct, os, numpy as np
H=5120; NK,NV,DK,DV,CK=16,48,128,128,4
KEY=NK*DK; VAL=NV*DV; CD=2*KEY+VAL
I=64
S=59

def bf16_bytes_to_f32(b):
    u=np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16
    return u.view(np.float32)
def f32_to_bf16(a):
    return (np.ascontiguousarray(a,dtype=np.float32).view(np.uint32)>>16).astype(np.uint16)

T={}
pre="model.language_model."
emb=bf16_bytes_to_f32(open('t/embed_rows.bin','rb').read()).reshape(S,H)
T[pre+"embed_tokens.weight"]=emb
T["lm_head.weight"]=np.zeros((64,H),np.float32)
T[pre+"norm.weight"]=np.ones(H,np.float32)
p=pre+"layers.0."
def real(fn,shape):
    return bf16_bytes_to_f32(open('t/'+fn,'rb').read()).reshape(*shape)
T[p+"input_layernorm.weight"]=real('input_layernorm.weight',[H])
T[p+"post_attention_layernorm.weight"]=np.ones(H,np.float32)
for nm,shape in [("in_proj_qkv.weight",[CD,H]),("in_proj_z.weight",[VAL,H]),("in_proj_b.weight",[NV,H]),("in_proj_a.weight",[NV,H]),("conv1d.weight",[CD,1,CK]),("dt_bias",[NV]),("A_log",[NV]),("norm.weight",[DV]),("out_proj.weight",[H,VAL])]:
    T[p+"linear_attn."+nm]=real('linear_attn.'+nm,shape)
T[p+"mlp.gate_proj.weight"]=np.zeros((I,H),np.float32)
T[p+"mlp.up_proj.weight"]=np.zeros((I,H),np.float32)
T[p+"mlp.down_proj.weight"]=np.zeros((H,I),np.float32)

import os
out=os.environ.get("OUT","../q35-l0-model")
os.makedirs(out,exist_ok=True)
header={}; off=0; blobs=[]
for name,a in T.items():
    b=f32_to_bf16(a).tobytes()
    header[name]={"dtype":"BF16","shape":list(a.shape),"data_offsets":[off,off+len(b)]}
    off+=len(b); blobs.append(b)
hj=json.dumps(header).encode()
with open(out+"/model.safetensors","wb") as f:
    f.write(struct.pack("<Q",len(hj))); f.write(hj)
    for b in blobs: f.write(b)
cfg={"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",
 "text_config":{"model_type":"qwen3_5_text","hidden_size":H,"intermediate_size":I,"num_hidden_layers":1,
  "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":64,"rms_norm_eps":1e-6,
  "attn_output_gate":True,"layer_types":["linear_attention"],"partial_rotary_factor":0.25,
  "rope_parameters":{"rope_theta":10000000,"rope_type":"default","partial_rotary_factor":0.25},
  "linear_num_key_heads":NK,"linear_num_value_heads":NV,"linear_key_head_dim":DK,"linear_value_head_dim":DV,
  "linear_conv_kernel_dim":CK,"tie_word_embeddings":False}}
json.dump(cfg,open(out+"/config.json","w"))
print("built", out)
# expected: torch res last token from arbiter (recompute quickly here in numpy — matches torch to 1e-6)
