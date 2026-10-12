"""Read the A733's 128-byte VNPD v2 records and map SDK node IDs.

The bundled SDK reader expects newer 192-byte records. The first eight
16-byte segments have the same format (HEAD, SEL0..5, SEL15). Counters are
hardware busy cycles, not additive end-to-end latency. Repeated compiler
UIDs are retained as candidate labels, never assigned to a guessed layer.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct


def read_npd(path):
    raw = path.read_bytes()
    magic, version, target, nets, cores = struct.unpack_from('<4sIIII', raw)
    assert (magic, version, nets, cores) == (b'VNPD', 2, 1, 1)
    assert target == 0x1000003b, 'counter layout is only verified on this A733 target'
    name, offset, size, count, stride, netcores = struct.unpack_from('<64sIIIII', raw, 256)
    assert stride in (128, 192) and netcores == cores
    assert size == count * stride and offset >= 384 and offset + size == len(raw)
    rows = []
    for i in range(count):
        start = offset + i * stride
        node, core = struct.unpack_from('<QQ', raw, start)
        layer, layer_over, instr, cycles, cycle_over = struct.unpack_from('<HHIII', raw, start + 16)
        reads, read_over, read_bytes, read_bytes_over = struct.unpack_from('<IIII', raw, start + 32)
        writes, write_over, write_bytes, write_bytes_over = struct.unpack_from('<IIII', raw, start + 48)
        mac_cycles, mac_over, mac_count, mac_count_over = struct.unpack_from('<IIII', raw, start + 80)
        assert core == 0 and not (layer_over & 0x8000)
        assert not any(x & 0x80000000 for x in (cycle_over, read_over, read_bytes_over,
                      write_over, write_bytes_over, mac_over, mac_count_over)), 'counter overflow'
        rows.append(dict(node=node, cycles=cycles, read_bytes=read_bytes,
                         write_bytes=write_bytes, mac_count=mac_count))
    return dict(name=name.rstrip(b'\0').decode(), target=hex(target), record_size=stride,
                sha256=hashlib.sha256(raw).hexdigest(), rows=rows)


def summarize(folder, npd):
    trace = json.loads((folder / 'export/node_trace_db.json').read_text())
    source = (folder / 'export/vnn_decoder.c').read_text()
    # Compiler UID suffixes repeat across fused decoder layers. Group only
    # when all corresponding source operations agree.
    labels = defaultdict(list)
    for lid, name, op, uid in re.findall(
        r'lid\s+- (\S+)\s+var\s+-[^\n]+\s+name\s+- (\S+)\s+operation\s+- (\S+)'
        r'.*?NEW_VXNODE\([^;]+, (\d+)\);', source, re.S):
        labels[int(uid)].append(dict(lid=lid, name=name, op=op))
    mapping = defaultdict(set)
    for t in trace:
        for n in t['profile_node_ids']:
            mapping[n['node_id']].add(t['uid']['id'])
    report = read_npd(npd)
    by_op = defaultdict(lambda: dict(cycles=0, read_bytes=0, write_bytes=0, mac_count=0))
    for row in report['rows']:
        candidates = [v for uid in mapping[row['node']] for v in labels[uid]]
        ops = sorted({v['op'] for v in candidates})
        row['op'] = '+'.join(ops) or 'unmapped'
        row['candidate_lids'] = sorted({v['lid'] for v in candidates})
        for key in by_op[row['op']]:
            by_op[row['op']][key] += row[key]
    report['summed_busy_cycles'] = sum(r['cycles'] for r in report['rows'])
    report['by_op'] = dict(sorted(by_op.items(), key=lambda v: -v[1]['cycles']))
    report['rows'].sort(key=lambda r: -r['cycles'])
    report['scope'] = 'Hardware busy counters; not end-to-end latency; compiler fusion and UID ambiguity retained'
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True, help='compiled graph with --node-trace')
    p.add_argument('--npd', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = summarize(a.folder, a.npd)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print(json.dumps(report['rows'][:12], indent=2))
