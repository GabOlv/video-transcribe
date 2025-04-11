import os
import subprocess
import sys
import whisper
import torch
from pathlib import Path
from tqdm import tqdm
from tkinter import Tk, filedialog
import re
import warnings

# === CONSTANTES ===
MODEL_SIZE = "medium"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === UTILITÁRIOS ===
warnings.filterwarnings("ignore")


def selecionar_arquivo():
    Tk().withdraw()
    print("Selecione um arquivo de vídeo ou áudio...")
    return filedialog.askopenfilename()


def get_basename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_duration(filepath):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filepath,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    return float(result.stdout.strip())


def run_ffmpeg_progress(cmd, input_path=None, desc="Processando"):
    total_duration = get_duration(input_path) if input_path else None
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")

    with tqdm(total=total_duration or 100, desc=desc, unit="s") as pbar:
        for line in process.stderr:
            match = time_pattern.search(line)
            if match:
                h, m, s, ms = map(int, match.groups())
                elapsed = h * 3600 + m * 60 + s + ms / 100
                if total_duration:
                    pbar.n = min(elapsed, total_duration)
                else:
                    pbar.update(1)
                pbar.refresh()
        pbar.n = total_duration or pbar.n
        pbar.refresh()
    process.wait()


def detectar_cuda():
    return torch.cuda.is_available()


# === ETAPAS DO PIPELINE ===


def extrair_audio(input_path, output_audio):
    print("Extraindo áudio...")
    run_ffmpeg_progress(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "44100",
            "-ac",
            "2",
            output_audio,
        ],
        input_path=input_path,
        desc="Extraindo áudio",
    )


def aprimorar_audio(input_audio, output_audio):
    print("Aprimorando áudio...")
    run_ffmpeg_progress(
        [
            "ffmpeg",
            "-i",
            input_audio,
            "-af",
            "highpass=f=200, lowpass=f=3000, afftdn=nf=-25, loudnorm",
            output_audio,
        ],
        input_path=input_audio,
        desc="Aprimorando áudio",
    )


def transcrever_audio(audio_path, legenda_path, use_cuda):
    print("Transcrevendo áudio com Whisper...")
    model = whisper.load_model(MODEL_SIZE, device="cuda" if use_cuda else "cpu")
    result = model.transcribe(
        audio_path,
        language="pt",
        task="transcribe",
        verbose=False,  # Evita logs detalhados
        fp16=use_cuda,  # Usa float16 só se for GPU
    )

    if "segments" not in result:
        print("Nenhuma transcrição encontrada.")
        return

    def format_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(legenda_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip().replace("-->", "→")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print("Legenda gerada com sucesso:", legenda_path)


# === MAIN ===


def main(filepath):
    if not filepath:
        print("Nenhum arquivo selecionado.")
        return

    use_cuda = detectar_cuda()
    print(
        "CUDA disponível. Usando GPU."
        if use_cuda
        else "CUDA não disponível. Usando CPU."
    )

    filename_base = get_basename(filepath)
    audio_enhanced = os.path.join(DATA_DIR, f"{filename_base}_enhanced.mp3")
    legenda_srt = os.path.join(DATA_DIR, f"{filename_base}_legenda.srt")

    is_video = any(
        filepath.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"]
    )

    if is_video:
        audio_temp = os.path.join(DATA_DIR, "temp_audio.mp3")
        extrair_audio(filepath, audio_temp)
        aprimorar_audio(audio_temp, audio_enhanced)
        os.remove(audio_temp)
    else:
        aprimorar_audio(filepath, audio_enhanced)

    transcrever_audio(audio_enhanced, legenda_srt, use_cuda)
    print("\n✅ Processo finalizado com sucesso!")
    print(f"Legenda: {legenda_srt}")
    print(f"Áudio melhorado: {audio_enhanced}")


if __name__ == "__main__":
    file_path = selecionar_arquivo()
    main(file_path)
