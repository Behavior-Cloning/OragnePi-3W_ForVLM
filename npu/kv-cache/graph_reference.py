"""Small FP32 interpreter for our imported decoder graph (no quant simulation)."""
import torch
import torch.nn.functional as F


def evaluate(graph, data, feeds):
    nodes=graph['Layers'];cache={}
    def visit(key):
        if key in cache:return cache[key]
        node=nodes[key];op=node['op'];p=node.get('parameters',{})
        x=[visit(v[1:].rsplit(':',1)[0]) for v in node.get('inputs',[])]
        if op=='input':y=feeds[node['name']]
        elif op=='variable':y=torch.from_numpy(data[key]['data'])
        elif op=='output':y=x[0]
        elif op=='fullconnect':y=x[0]@torch.from_numpy(data[key]['weight'])+torch.from_numpy(data[key]['bias'])
        elif op=='reshape':y=x[0].reshape(p['shape'])
        elif op=='permute':y=x[0].permute(p['perm'])
        elif op=='slice':y=x[0][tuple(slice(b,b+s) for b,s in zip(p['begin'],p['size']))]
        elif op=='concat':y=torch.cat(x,dim=p['dim'])
        elif op=='tile':y=x[0].repeat(p['multiples'])
        elif op=='matmul':y=x[0]@x[1]
        elif op=='add':y=x[0]+x[1]
        elif op=='multiply':y=x[0]*x[1]
        elif op in ('divide','real_div'):y=x[0]/x[1]
        elif op=='neg':y=-x[0]
        elif op=='abs':y=x[0].abs()
        elif op=='sqrt':y=x[0].sqrt()
        elif op=='swish':y=F.silu(x[0])
        elif op=='softmax':y=torch.softmax(x[0],dim=p['sf_axis'])
        elif op=='reducemax':y=x[0].amax(tuple(p['axis_list']),keepdim=p['keep_dims'])
        elif op=='reducemean':y=x[0].mean(tuple(p['axis_list']),keepdim=p['keep_dims'])
        elif op=='clipbyvalue':y=x[0].clamp(p['clip_value_min'],p['clip_value_max'])
        else:raise ValueError(op)
        assert torch.isfinite(y).all(),key
        cache[key]=y
        return y
    return {node['name']:visit(key) for key,node in nodes.items() if node['op']=='output'}
