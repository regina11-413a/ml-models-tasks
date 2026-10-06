from transformers import pipeline
import librosa
import numpy as np


print("Загрузка модели распознавания речи...")
asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h"
)
print("Модель загружена.")


def load_audio(file_path: str, target_sr: int = 16000) -> dict:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return {"array": audio.astype(np.float32), "sampling_rate": sr}


def transcribe(audio_path: str) -> str:
    audio_input = load_audio(audio_path)
    result = asr(
        audio_input["array"],
        sampling_rate=audio_input["sampling_rate"],
        chunk_length_s=30
    )
    return result["text"]


if __name__ == "__main__":
    AUDIO_FILE = "C:/Users/regin/PycharmProjects/ml-models-tasks/samples/sample-speech-1m.wav"

    print("=" * 60)
    print("РАСПОЗНАВАНИЕ РЕЧИ (SPEECH-TO-TEXT)")
    print("=" * 60)

    try:
        text = transcribe(AUDIO_FILE)
        print(f"\nРаспознанный текст:\n{text}")
    except FileNotFoundError:
        print(f"Файл не найден: {AUDIO_FILE}")
    except Exception as e:
        print(f"Ошибка: {e}")