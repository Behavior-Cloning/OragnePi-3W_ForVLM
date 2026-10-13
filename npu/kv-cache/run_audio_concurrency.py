"""Compile and schedule real resident ASR against matched CPU/NPU fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from board_host_lock import lock_board_from_host


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--cpu',type=Path,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--audio-setup',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    scripts=Path(__file__).resolve().parent
    state={'status':'waiting-audio-setup','scope':'paired resident ASR effect; full-depth 512+64 only'}
    def save():
        tmp=a.output/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'state.json')
    save()
    deadline=time.monotonic()+14400
    while True:
        setup_path=a.audio_setup/'state.json'
        setup=json.loads(setup_path.read_text()) if setup_path.exists() else {}
        assert setup.get('status')!='failed', 'audio setup failed; inspect it before concurrency'
        if setup.get('status')=='passed-setup-only':break
        assert time.monotonic()<deadline, 'audio setup did not finish'
        time.sleep(10)
    ref=a.reference.resolve()
    m=json.loads((ref.parents[1]/'trained-manifest.json').read_text())
    r=json.loads((ref/'tokens.json').read_text())
    cpu=json.loads(a.cpu.read_text())
    assert cpu['status']=='complete'
    assert r['decoder_layers']==m['language_layers_total'] and r['head_on_npu']
    assert r['fixed_feedback'] and r['generated_tokens']==128 and r['cached_prefix_tokens']==512 and r['new_prompt_tokens']==64
    assert json.loads((ref/'full-validation-w8.json').read_text())['execution_gate_v1']['passed']
    assert cpu['nonstream_trajectory']['tokens']==r['feedback_tokens']
    prompt=json.loads((ref/'prompt.json').read_text())['token_ids']
    assert cpu['prompt']['token_ids']==prompt and cpu['cached_prefix']==512
    state.update(status='waiting-board',reference=str(ref),cpu_reference=str(a.cpu),
                 source_sha256={n:sha(scripts/n) for n in ('audio_concurrency_board.py','resident_session.c','resident_worker.py',
                     'full_model.c','batch_prefill.h','resource_sampler.py','asr-worker/asr_worker.cpp')})
    save()
    lock_board_from_host()
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
    scp=prefix+['scp','-o','ForwardX11=no','-o','ConnectTimeout=8']
    assert all(x.replace('-','').isalnum() for x in (a.tag,a.output.name))
    remote='/opt/opi-vlm/npu-kv-probe/'+a.output.name
    audio_root=setup['remote']
    config={'remote':remote,'deployed':'/opt/opi-vlm/npu-kv-probe/'+a.tag,
            'prompt':prompt,'feedback':r['feedback_tokens'],'expected_tokens':r['tokens'],
            'cached_prefix':512,'capacity':r['capacity'],'layers':r['decoder_layers'],
            'vocab':m['vocab_size'],'eos':m['eos_token_ids'][0],'batch':r['prefill_batch_size'],
            'expected_logits_sha256':sha(ref/'logits.f32'),'cpu_command':cpu['command'],
            'cpu_reference_sha256':sha(a.cpu),'cpu_scope':'matched Q8 configuration; see separate fastest CPU search',
            'asr_binary':remote+'/asr-build/opi-asr-worker','asr_model':audio_root+'/ggml-tiny-q5_1.bin',
            'asr_wav':audio_root+'/whisper.cpp/samples/jfk.wav','asr_model_sha256':setup['model_sha256'],
            'asr_source_commit':setup['source_commit']}
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    try:
        subprocess.run(ssh+['mkdir '+shlex.quote(remote)],check=True,timeout=30)
        files=['audio_concurrency_board.py','resident_worker.py','resident_session.c','full_model.c','board_probe.c',
               'batch_prefill.h','greedy_i16.h','resource_sampler.py','thermal_condition.py','thermal_heater.c']
        subprocess.run(scp+['-r',*[str(scripts/n) for n in files],str(scripts/'asr-worker'),str(a.output/'config.json'),
                            'root@192.168.88.29:'+remote+'/'],check=True,timeout=90)
        manifest_flags=[f'-DMODEL_DIM={m["dim"]}',f'-DMODEL_KV={m["kv_heads"]}',f'-DMODEL_HD={m["head_dim"]}',
                        f'-DROPE_BASE={m["rope_base"]}',f'-DFUSED_LAYERS={r["fusion_width"]}',
                        f'-DKEY_CACHE_TRANSPOSED={int(r["key_cache_transposed"])}',
                        f'-DHEAD_SHARDS={len(m.get("head_shards",[{}]))}',
                        f'-DHEAD_CHUNK={m.get("head_shards",[{"vocab_count":m["vocab_size"]}])[0]["vocab_count"]}']
        commands=[['cmake','-S',remote+'/asr-worker','-B',remote+'/asr-build','-DCMAKE_BUILD_TYPE=Release',
                   '-DWHISPER_SOURCE='+audio_root+'/whisper.cpp'],
                  ['nice','-n','10','taskset','-c','0,1','cmake','--build',remote+'/asr-build','--target','opi-asr-worker','-j','2'],
                  ['cc','-O2','-Wall','-Wextra','-Werror',*manifest_flags,'resident_session.c','-lNBGlinker','-lVIPhal','-lm','-o','resident'],
                  ['cc','-O3','-Wall','-Wextra','-Werror','-march=armv8-a+simd','thermal_heater.c','-pthread','-o','thermal_heater'],
                  ['python3','-u','audio_concurrency_board.py',remote+'/config.json']]
        command='cd '+shlex.quote(remote)+' && '+' && '.join(shlex.join(c) for c in commands)
        state['status']='board-running';save()
        with (a.output/'run.log').open('x') as log:
            run=subprocess.run(ssh+['flock -w 1800 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command)],
                               stdout=log,stderr=subprocess.STDOUT,timeout=5400)
        for name in ('result.json','npu.log','cpu.log','asr.log'):
            subprocess.run(scp+['root@192.168.88.29:'+remote+'/'+name,str(a.output/name)],timeout=60)
        run.check_returncode()
        result=json.loads((a.output/'result.json').read_text())
        assert result['status']=='complete-paired-concurrency'
        state.update(status='complete-paired-concurrency',medians=result['medians']);save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
