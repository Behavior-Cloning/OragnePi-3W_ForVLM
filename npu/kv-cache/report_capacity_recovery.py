"""Archive the exact recoverable cleanup performed after board ENOSPC."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results'


def main():
    evidence={}
    for name in ('graph-data-dedup-all-20261013','board-model-dedup-20261013',
                 'verified-logits-prune-second-20261013','archived-board-dumps-prune-20261013'):
        path=ROOT/(name+'.json');raw=path.read_bytes();data=json.loads(raw)
        assert data.get('status','complete')=='complete'
        evidence[name]={**data,'source':str(path),'source_sha256':hashlib.sha256(raw).hexdigest()}
    archive=HERE/'evidence-attack-20261013/capacity-recovery.json'
    archive.write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['# TF容量恢复与可恢复归档','',
           '2026-10-13约16:04，板端58GiB根文件系统实际可用为0，mkdir明确返回No space left on device。部分诊断/补测停止，原日志和失败attempt保留；网页当时仍ready，温度约38–40°C。这次容量不足不能称为温度导致掉卡。','',
           '| 操作 | 实际处理 | 数据保留方式 |','| --- | --- | --- |',
           '| 主机decoder.data去重 | 释放59.91GiB | SHA256相同者硬链接，原路径及内容全部保留 |',
           '| 板端NBG/embedding去重 | 77个重复文件，释放2.72GiB | 原路径及内容全部保留，未删除模型路径 |',
           '| 第二批已完成A/B诊断副本 | 移除640.60MiB | 本机完整logits备份逐个SHA256一致 |',
           '| 板端历史logits/hidden副本 | 238份，合计8.08GiB | 本机每个文件完整复核后才移除板端副本 |','',
           '最后一步结束时，板端可用11408064512字节，约10.62GiB。之后上传新候选会继续占用，不把这个快照当成实时余量。CPU网页、模型权重、编译图路径和结果JSON均保留；没有格式化TF或关机。','',
           '恢复已移除的副本：证据中的每条`remote`是原板端文件，`backup`是本机完整备份，`sha256`和`bytes`用于校验。先获取共同实验锁并确认空间，再把该backup复制回对应remote；不要对整个工作目录执行删除或覆盖。NBG硬链接去重无须恢复，因为文件名和字节未变。','',
           'Git保留清单、代码和小型结果；大logits、模型和SDK仍只在本机忽略目录，Git备份并不包含这些大文件本身。','',
           '续跑使用新attempt，不覆写失败证据。部署器检查新增上传/诊断大小并预留256MiB；性能runner在开始时预留全部完整logits及512MiB。已有19项本地验证和两模型resident实板检查通过，容量恢复不等于长期稳定性验收。','',
           '[完整校验与恢复清单](evidence-attack-20261013/capacity-recovery.json)。','']
    (HERE/'TF容量恢复与归档.md').write_text('\n'.join(lines))
    print('recovery manifest archived')


if __name__=='__main__':main()
