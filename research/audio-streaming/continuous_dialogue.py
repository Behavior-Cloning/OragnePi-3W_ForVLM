import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import sherpa_onnx as sherpa

sys.path.insert(0, "/opt/opi-vlm/audio-lab-20261018")
import audio_worker as aw
sys.path.insert(0, "/opt/opi-vlm/asr-small-20261018")
import worker as sw

BASE = Path("/opt/opi-vlm/audio-streaming-20261018")
KWS = BASE / "kws"
CORPUS = Path("/opt/opi-vlm/asr-small-20261018/corpus")
MANIFEST = json.loads((CORPUS / "manifest.json").read_text())
SAMPLES = MANIFEST["samples"]
LLM = "http://127.0.0.1:8081/v1/chat/completions"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
RATE = 16000
TURN_IDS = [0, 8, 16]
WAKE = KWS / "candidate/tts/pos-alone-normal.wav"


def snapshot():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:")):
            fields = line.split()
            mem[fields[0].rstrip(":")] = int(fields[1])
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
            status[line.split(":")[0]] = int(line.split()[1])
    llama = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_text(errors="ignore")
            if "llama-server" in cmdline:
                for line in (proc / "status").read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
                        llama[line.split(":")[0]] = int(line.split()[1])
                llama["pid"] = int(proc.name)
                break
        except OSError:
            pass
    return {
        "monotonic": time.monotonic(),
        "temperature_mC": int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()),
        "meminfo_kib": mem,
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "cpu_ticks": Path("/proc/stat").read_text().splitlines()[0],
        "llama_server": llama,
        "self": status,
    }


def read_wav(path):
    import wave

    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == RATE and f.getnchannels() == 1 and f.getsampwidth() == 2
        return np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32) / 32768


def make_kws():
    model = KWS / "model"
    return sherpa.KeywordSpotter(
        tokens=str(model / "tokens.txt"),
        encoder=str(model / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),
        decoder=str(model / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
        joiner=str(model / "joiner-epoch-13-avg-2-chunk-16-left-64.onnx"),
        keywords_file=str(KWS / "candidate/keywords.txt"),
        num_threads=1,
        keywords_threshold=0.25,
    )


def detect_wake(spotter, samples):
    stream = spotter.create_stream()
    started = time.monotonic()
    for i in range(0, len(samples), 1280):
        chunk = samples[i : i + 1280]
        due = started + (i + len(chunk)) / RATE
        time.sleep(max(0, due - time.monotonic()))
        stream.accept_waveform(RATE, chunk)
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            return {
                "triggered": True,
                "keyword": result,
                "latency_s": time.monotonic() - started,
                "audio_s": (i + len(chunk)) / RATE,
            }
    return {
        "triggered": False,
        "keyword": spotter.get_result(stream),
        "latency_s": time.monotonic() - started,
        "audio_s": len(samples) / RATE,
    }


def llm_stream(messages, max_tokens=48):
    body = json.dumps(
        {
            "model": "qwen25-15b-q4",
            "messages": messages,
            "stream": True,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        LLM, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    first = None
    parts = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            piece = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if piece:
                if first is None:
                    first = time.monotonic() - start
                parts.append(piece)
    return {"first_token_s": first, "complete_s": time.monotonic() - start, "text": "".join(parts)}


def monitor(stop, records):
    while not stop.wait(0.25):
        records.append(snapshot())


def run(profile):
    denoise_on = profile == "full"
    cpus, threads = ({2, 3}, 1) if denoise_on else ({6, 7}, 2)
    os.sched_setaffinity(0, cpus)
    denoiser = aw.model("denoise", 1) if denoise_on else None
    vad = aw.model("vad", 1, 0.5)
    asr = sw.model("para", threads)
    spotter = make_kws()
    warm = asr.create_stream()
    warm.accept_waveform(RATE, read_wav(CORPUS / SAMPLES[TURN_IDS[0]]["file"]))
    asr.decode_stream(warm)
    if denoiser:
        denoiser.reset()

    system_samples = []
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop, system_samples), daemon=True)
    thread.start()
    messages = [{"role": "system", "content": f"你是家庭陪伴机器人，结合此前对话，用简洁中文回答用户。会话标记：{profile}。"}]
    turns = []
    try:
        for n, sid in enumerate(TURN_IDS, 1):
            row = SAMPLES[sid]
            wake_start = time.monotonic()
            wake = detect_wake(spotter, read_wav(WAKE))
            if not wake["triggered"]:
                raise RuntimeError("wake word did not trigger")

            vad.reset()
            if denoiser:
                denoiser.reset()
            pending = np.zeros(0, np.float32)
            segments = []
            frames_ms = []
            endpoint_time = None
            speech_begin = time.monotonic()
            command = read_wav(CORPUS / row["file"])
            playback = np.concatenate([command, np.zeros(int(0.8 * RATE), np.float32)])
            for offset in range(0, len(playback), 256):
                block = playback[offset : offset + 256]
                due = speech_begin + (offset + len(block)) / RATE
                time.sleep(max(0, due - time.monotonic()))
                t0 = time.perf_counter()
                clean = np.asarray(denoiser(block, RATE).samples, np.float32) if denoiser else block
                frames_ms.append((time.perf_counter() - t0) * 1000)
                pending = np.concatenate([pending, clean])
                while len(pending) >= 512:
                    vad.accept_waveform(pending[:512])
                    pending = pending[512:]
                    while not vad.empty():
                        seg = vad.front
                        segments.append(np.asarray(seg.samples, np.float32).copy())
                        vad.pop()
                        endpoint_time = time.monotonic()
                if segments:
                    break
            if len(pending):
                vad.accept_waveform(np.pad(pending, (0, 512 - len(pending))))
                while not vad.empty():
                    seg = vad.front
                    segments.append(np.asarray(seg.samples, np.float32).copy())
                    vad.pop()
                    endpoint_time = time.monotonic()
            vad.flush()
            while not vad.empty():
                seg = vad.front
                segments.append(np.asarray(seg.samples, np.float32).copy())
                vad.pop()
                endpoint_time = time.monotonic()
            if not segments:
                raise RuntimeError("VAD returned no segment")

            endpoint_time = endpoint_time or time.monotonic()
            asr_start = time.monotonic()
            texts = []
            for segment in segments:
                stream = asr.create_stream()
                stream.accept_waveform(RATE, segment)
                asr.decode_stream(stream)
                if stream.result.text:
                    texts.append(stream.result.text)
            asr_end = time.monotonic()
            recognized = "".join(texts)
            messages.append({"role": "user", "content": recognized})
            answer = llm_stream(messages)
            messages.append({"role": "assistant", "content": answer["text"]})
            frames_ms.sort()
            turns.append(
                {
                    "turn": n,
                    "source_id": row["id"],
                    "reference": row["text"],
                    "asr_text": recognized,
                    "wake_to_trigger_s": wake["latency_s"],
                    "wake_keyword": wake["keyword"],
                    "wake_to_vad_endpoint_s": endpoint_time - wake_start,
                    "command_start_to_vad_endpoint_s": endpoint_time - speech_begin,
                    "endpoint_to_asr_complete_s": asr_end - endpoint_time,
                    "wake_to_asr_complete_s": asr_end - wake_start,
                    "asr_to_llm_first_token_s": answer["first_token_s"],
                    "asr_to_llm_complete_s": answer["complete_s"],
                    "wake_to_first_token_s": asr_end - wake_start + answer["first_token_s"],
                    "wake_to_reply_complete_s": time.monotonic() - wake_start,
                    "denoise": denoise_on,
                    "asr_threads": threads,
                    "asr_affinity": sorted(cpus),
                    "frontend_frame_p95_ms": frames_ms[max(0, int(0.95 * len(frames_ms)) - 1)],
                    "frontend_frames_over_16ms": sum(x > 16 for x in frames_ms),
                    "segments": len(segments),
                    "llm_reply": answer["text"],
                    "llm_history_messages": len(messages),
                    "resources": snapshot(),
                }
            )
            print(json.dumps(turns[-1], ensure_ascii=False), flush=True)
    finally:
        stop.set()
        thread.join()
    return {
        "profile": profile,
        "denoise": denoise_on,
        "asr_threads": threads,
        "turns": turns,
        "system_samples": system_samples,
        "end": snapshot(),
    }


llm_stream(
    [
        {"role": "system", "content": "服务预热，不要复用到测试会话。"},
        {"role": "user", "content": "请简短回答：准备好了么？"},
    ],
    max_tokens=4,
)
results = []
for profile in ("full", "fast"):
    print("PROFILE", profile, flush=True)
    results.append(run(profile))
(OUT / "summary.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "scope": "3-turn replay per profile; wake phrase is public synthetic TTS, commands are public AISHELL clips; LLM is resident CPU llama-server Qwen2.5-1.5B Q4; not live microphone or NPU-backed dialogue",
            "profiles": results,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
