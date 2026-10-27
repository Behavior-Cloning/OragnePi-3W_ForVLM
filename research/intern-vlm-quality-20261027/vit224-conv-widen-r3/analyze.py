from pathlib import Path
import json,hashlib,statistics,numpy as np
P=Path(__file__).resolve().parent;s=json.loads((P/'results/summary.json').read_text());assert s['status']=='complete';audit=[]
for r in s['records']:
 f=P/'results'/(r['name']+'.f32');x=np.fromfile(f,dtype='<f4');ref=np.fromfile(P/f"l{r['layer']:02d}-reference{r['image']}.f32",dtype='<f4');base=np.fromfile(P/'results'/f"l{r['layer']:02d}-base-before-i{r['image']}.f32",dtype='<f4')
 assert np.isfinite(x).all() and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];assert r['metrics']['released_bytes']==0 and r['metrics']['differing_elements']==0
 audit.append({'name':r['name'],'layer':r['layer'],'mode':r['mode'],'image':r['image'],'api_ms':statistics.median(r['metrics']['api_seconds'])*1000,'driver_bytes':r['metrics']['driver_bytes'],'rel_l2_fp32':float(np.linalg.norm(x-ref)/np.linalg.norm(ref)),'rel_l2_base':float(np.linalg.norm(x-base)/np.linalg.norm(base)),'same_bytes_as_base':x.tobytes()==base.tobytes(),'sha256':r['sha256'],'released_bytes':0})
(P/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
lines=['# '+('W8激活降位筛查：淘汰' if 'a8-screen' in P.name else '等价1×1映射：输出相同但更慢，不晋升'),'', '真实板端NPU单层实验；每配置2张参考激活，2预热+10热API调用，CPU7，baseline-before → candidate → baseline-after。保留Transformer网络语义；不是整塔、端到端、VQA或decode测试，无system/dynamic文字token。单层诊断F32读回单列，不当作部署流水线开销。','', '|层/配置/图|NPU API中位ms|相对FP32 L2|与基线同字节|驱动MiB|','|---|---:|---:|---|---:|']
for q in audit:lines.append(f"|{q['name']}|{q['api_ms']:.3f}|{q['rel_l2_fp32']*100:.3f}%|{q['same_bytes_as_base']}|{q['driver_bytes']/2**20:.2f}|")
if 'a8-screen' in P.name:lines+=['', '在层0和12将FC输入（及另一版FC输出）由I16 DFP降低为I8，缩小小数位以保留相同物理范围。权重W8、敏感norm/GELU FP16保持；未使用额外校准/训练。候选约60–65ms，基线约46–47ms，且相对FP32误差最高约33%。结果说明此直接降位方案不可用，不能推断所有W8A8训练模型都不可用。未扩展24层，也未覆盖默认。']
else:lines+=['', '先前A16DFP+W8PC卷积在SDK拒绝，I16DFP→I16affine转换也未注册。此版将原W8整数码用I16容器存储，原逐通道scale不变；6个FC的整数码范围[-127,127]、与原W8逐元素一致，reshape为1×1卷积。不是改变LLM/ViT架构，也没有新增模型知识。权重文件的实数值不变；FP32数学等价测试误差约1e-7。','', 'SDK编译通过且所测单层两图输出均与原版逐字节相同，但NPU API约50.84/51.04ms，比原版46.5–46.7ms慢约9%；驱动约17.68→25.01MiB。无收益，不扩展24层、不晋升默认。局部输出一致不能代替全塔VQA认证。']
lines+=['', '全部输出finite、重复稳定、每进程关闭driver0。SHA重新从回收f32计算。FP32参考仅用于误差诊断，实际测速与执行都在NPU；不把FP32主机诊断当NPU推理速度。证据：compile-status.json、config.json、results/summary.json、AUDIT.json；权重/NBG/f32保留Windows磁盘，Git只保存来源、脚本、摘要和哈希。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(P.name,len(audit))
