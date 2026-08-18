import io
import os
import time
import torch
import boto3
import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal
import torchaudio
import requests
import runpod
from qwen_tts import Qwen3TTSModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Initializing Qwen3-TTS Models on {DEVICE} ({DTYPE})...")
model_base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=DEVICE, dtype=DTYPE)
model_design = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map=DEVICE, dtype=DTYPE)
print("Base and VoiceDesign Models Loaded in VRAM.")

s3_client = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ.get('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
)
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_PUBLIC_BASE_URL = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")


def apply_production_mastering(audio_data, sample_rate, target_lufs=-16.0):
    audio = audio_data.astype(np.float32)
    b, a = signal.butter(4, 80.0 / (sample_rate / 2.0), btype="high")
    audio_filtered = signal.filtfilt(b, a, audio)
    peak = np.max(np.abs(audio_filtered))
    if peak > 0.85:
        audio_filtered = audio_filtered * (0.85 / peak)
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio_filtered)
    if current_lufs > -70.0:
        normalized = pyln.normalize.loudness(audio_filtered, current_lufs, target_lufs)
    else:
        normalized = audio_filtered
    max_peak = np.max(np.abs(normalized))
    if max_peak > 0.94:
        normalized = normalized * (0.94 / max_peak)
    return normalized


def upload_audio(audio_data, sr, object_key):
    wav_tensor = torch.from_numpy(audio_data).unsqueeze(0)
    mp3_buffer = io.BytesIO()
    torchaudio.save(mp3_buffer, wav_tensor, sr, format="mp3")
    mp3_buffer.seek(0)
    s3_client.upload_fileobj(mp3_buffer, R2_BUCKET, object_key, ExtraArgs={"ContentType": "audio/mpeg"})
    return f"{R2_PUBLIC_BASE_URL}/{object_key}"


def upload_tensor(tensor_data, object_key):
    buffer = io.BytesIO()
    torch.save(tensor_data, buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, R2_BUCKET, object_key)
    return object_key


def handler(job):
    job_input = job["input"]
    task = job_input.get("task", "render")
    job_id = job_input.get("job_id", f"job_{int(time.time())}")
    t_start = time.time()

    if task == "design_voice":
        instruct = job_input["instruct"]
        anchor_text = job_input["anchor_text"]
        target_voice_key = job_input.get("target_voice_key", f"voices/{job_id}.pt")

        with torch.inference_mode():
            wavs_anchor, sr = model_design.generate_voice_design(
                text=anchor_text, language="English", instruct=instruct, temperature=0.65
            )
            prompt_items = model_design.create_voice_clone_prompt(
                ref_audio=(wavs_anchor[0], sr), ref_text=anchor_text, x_vector_only_mode=False
            )

        upload_tensor(prompt_items, target_voice_key)
        mastered_audio = apply_production_mastering(wavs_anchor[0], sr, target_lufs=-16.0)
        audio_url = upload_audio(mastered_audio, sr, f"renders/{job_id}_preview.mp3")

        return {
            "status": "success",
            "task": task,
            "voice_prompt_key": target_voice_key,
            "preview_audio_url": audio_url,
        }

    elif task == "clone_voice":
        ref_audio_url = job_input["ref_audio_url"]
        ref_text = job_input["ref_text"]
        preview_text = job_input["preview_text"]
        target_voice_key = job_input.get("target_voice_key", f"voices/{job_id}.pt")

        response = requests.get(ref_audio_url)
        ref_audio_path = f"/tmp/{job_id}_ref.wav"
        with open(ref_audio_path, "wb") as f:
            f.write(response.content)

        with torch.inference_mode():
            prompt_items = model_base.create_voice_clone_prompt(
                ref_audio=ref_audio_path, ref_text=ref_text, x_vector_only_mode=False
            )
            wavs_preview, sr = model_base.generate_voice_clone(
                text=[preview_text], language="English", voice_clone_prompt=prompt_items, temperature=0.70
            )

        upload_tensor(prompt_items, target_voice_key)
        mastered_audio = apply_production_mastering(wavs_preview[0], sr, target_lufs=-16.0)
        audio_url = upload_audio(mastered_audio, sr, f"renders/{job_id}_preview.mp3")

        return {
            "status": "success",
            "task": task,
            "voice_prompt_key": target_voice_key,
            "preview_audio_url": audio_url,
        }

    elif task == "render":
        voice_prompt_key = job_input["voice_prompt_key"]
        script_json = job_input["script_json"]
        batch_size = job_input.get("batch_size", 8)

        voice_bytes = io.BytesIO()
        s3_client.download_fileobj(R2_BUCKET, voice_prompt_key, voice_bytes)
        voice_bytes.seek(0)
        prompt_items = torch.load(voice_bytes, map_location=DEVICE)

        total_acts = len(script_json)
        audio_segments = [None] * total_acts
        sr = 24000

        with torch.inference_mode():
            for batch_start in range(0, total_acts, batch_size):
                batch_end = min(batch_start + batch_size, total_acts)
                batch_items = script_json[batch_start:batch_end]

                texts = [item["processed_text"] for item in batch_items]
                temp = batch_items[0].get("recommended_temperature", 0.70)

                torch.manual_seed(42 + batch_start)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42 + batch_start)

                wavs, sr = model_base.generate_voice_clone(
                    text=texts, language="English", voice_clone_prompt=prompt_items, temperature=temp
                )

                for i, wav in enumerate(wavs):
                    audio_segments[batch_start + i] = wav

        stitched_chunks = []
        for wav in audio_segments:
            stitched_chunks.append(wav)
            silence = np.zeros(int(sr * 0.50), dtype=wav.dtype)
            stitched_chunks.append(silence)

        raw_master = np.concatenate(stitched_chunks)
        mastered_audio = apply_production_mastering(raw_master, sr, target_lufs=-16.0)
        audio_url = upload_audio(mastered_audio, sr, f"renders/{job_id}_master.mp3")

        total_duration_min = (len(raw_master) / sr) / 60.0
        render_time_sec = time.time() - t_start

        return {
            "status": "success",
            "task": task,
            "audio_url": audio_url,
            "audio_duration_minutes": round(total_duration_min, 2),
            "gpu_render_time_seconds": round(render_time_sec, 2),
        }


runpod.serverless.start({"handler": handler})
