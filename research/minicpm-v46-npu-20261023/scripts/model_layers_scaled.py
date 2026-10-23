from model_layers import *
class ScaledLinearLayer(torch.nn.Module):
 def __init__(self,layer):super().__init__();self.layer=layer
 def forward(self,hidden,conv,state):
  l=self.layer;a=l.linear_attn;x=stable_norm(hidden,l.input_layernorm.weight)
  qkv=a.in_proj_qkv(x).transpose(1,2)
  new_conv=torch.cat([conv[:,:,1:],qkv],dim=-1)
  mixed=torch.nn.functional.silu(sum(new_conv[:,:,j]*a.conv1d.weight[:,0,j] for j in range(4)))
  q,k,v=[t.reshape(1,16,1,128) for t in mixed.split(2048,dim=-1)]
  q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*(128**-0.5)
  k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
  g=-a.A_log.exp().reshape(1,16,1,1)*torch.nn.functional.softplus(a.in_proj_a(x)+a.dt_bias).reshape(1,16,1,1)
  beta=a.in_proj_b(x).sigmoid().reshape(1,16,1,1)
  decayed=state*torch.exp(g);delta=(v-k@decayed)*beta;new_state=decayed+k.transpose(-1,-2)@delta
  out=((q*32)@new_state);z=a.in_proj_z(x).reshape(1,16,1,128)
  out=((out*torch.rsqrt((out*out).mean(-1,keepdim=True)+0.001024)*a.norm.weight)*torch.nn.functional.silu(z)).reshape(1,1,2048);h=hidden+a.out_proj(out)
  return h+l.mlp(stable_norm(h,l.post_attention_layernorm.weight)),new_conv,new_state

LinearLayer=ScaledLinearLayer
