#!/usr/bin/env python3


import subprocess
import time
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Настройки
OUTPUT_DIR = "data/musiccaps_audio"
SAMPLE_RATE = 32000  
MAX_RETRIES = 3      
DELAY_BETWEEN = 3    

def download_clip(ytid: str, start_s: float, end_s: float, output_path: Path) -> bool:
    """
    Скачивает аудиофрагмент с YouTube через yt-dlp + ffmpeg.
    
    Args:
        ytid: YouTube video ID
        start_s: начало фрагмента в секундах
        end_s: конец фрагмента в секундах
        output_path: куда сохранить .wav файл
    
    Returns:
        True если успешно, False если ошибка
    """
    if output_path.exists():
        return True  # Уже скачано
    
   
    duration = end_s - start_s
    

    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",  # лучшее качество
        "--download-sections", f"*{start_s}-{end_s}",
        "--no-warnings",
        "--quiet",
        "-o", str(output_path),
        f"https://www.youtube.com/watch?v={ytid}"
    ]
    
   
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  
            )
            if result.returncode == 0 and output_path.exists():
                return True
        except subprocess.TimeoutExpired:
            pass  # Пробуем снова
        except Exception as e:
            print(f"Ошибка: {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt) 
    

    if output_path.exists():
        output_path.unlink()
    return False


def main():
    print("Загрузка датасета MusicCaps...")
    
    # Загружаем датасет с HuggingFace
    dataset = load_dataset("google/MusicCaps", split="train")
    print(f"Загружено {len(dataset)} примеров")
    
    # Создаём папку для аудио
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Счётчики для статистики
    success = 0
    failed = 0
    skipped = 0
    
    print(f"Начинаем скачивание в {output_dir}...")
    
    # Проходим по всем примерам
    for i, example in enumerate(tqdm(dataset, desc="Скачивание")):
        ytid = example["ytid"]
        start_s = example["start_s"]
        end_s = example["end_s"]
        
        output_path = output_dir / f"{ytid}.wav"
        
        # Пропускаем, если уже есть
        if output_path.exists():
            skipped += 1
            continue
        
        # Пытаемся скачать
        if download_clip(ytid, start_s, end_s, output_path):
            success += 1
        else:
            failed += 1
            # Сохраняем информацию о неудаче, чтобы не пробовать снова
            fail_log = output_dir / "failed.jsonl"
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ytid": ytid, "reason": "download_failed"}) + "\n")
        
        # Пауза чтобы не забанили
        time.sleep(DELAY_BETWEEN)
        
        # Печатаем прогресс каждые 100 примеров
        if (i + 1) % 100 == 0:
            print(f"\n Прогресс: {i+1}/{len(dataset)} |  {success} |  {failed} |  {skipped}")
    
    # Итоговый отчёт
    print("\n" + "="*50)
    print("ГОТОВО!")
    print(f" Успешно скачано: {success}")
    print(f" Не удалось: {failed}")
    print(f" Уже было: {skipped}")
    print(f" Файлы сохранены в: {output_dir.absolute()}")
    print("="*50)


if __name__ == "__main__":
    main()