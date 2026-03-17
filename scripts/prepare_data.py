import json
import random
import gzip
import shutil
from pathlib import Path

# Настройки
BASE_PATH = Path("data/musiccaps_complete")
MANIFEST_FILE = BASE_PATH / "manifest.jsonl"
TRAIN_FILE = BASE_PATH / "train.jsonl"
VALID_FILE = BASE_PATH / "valid.jsonl"
TRAIN_GZ = BASE_PATH / "train.jsonl.gz"
VALID_GZ = BASE_PATH / "valid.jsonl.gz"

print("="*60)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ AUDIOCRAFT")
print("="*60)

# 1. Проверка наличия манифеста
if not MANIFEST_FILE.exists():
    print(f" Файл {MANIFEST_FILE} не найден!")
    print("Сначала выполните часть 1.2 для создания манифеста")
    exit(1)

# 2. Чтение всех записей
print(f"Чтение {MANIFEST_FILE}...")
with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Найдено записей: {len(lines)}")

# 3. Перемешивание
random.seed(42)  # для воспроизводимости
random.shuffle(lines)

# 4. Разделение 90/10
split_idx = int(len(lines) * 0.9)
train_lines = lines[:split_idx]
valid_lines = lines[split_idx:]

print(f"\nРазделение данных:")
print(f"   Train: {len(train_lines)} файлов ({len(train_lines)/len(lines)*100:.1f}%)")
print(f"   Valid: {len(valid_lines)} файлов ({len(valid_lines)/len(lines)*100:.1f}%)")

# 5. Сохранение в обычные JSONL
print(f"\nСохранение JSONL файлов...")
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)
print(f"   {TRAIN_FILE}")

with open(VALID_FILE, 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)
print(f"    {VALID_FILE}")

# 6. Сжатие в GZ (требуется AudioCraft)
print(f"\n Сжатие в GZ формат...")

with open(TRAIN_FILE, 'rb') as f_in:
    with gzip.open(TRAIN_GZ, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"    {TRAIN_GZ}")

with open(VALID_FILE, 'rb') as f_in:
    with gzip.open(VALID_GZ, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"    {VALID_GZ}")

# 7. Проверка
print(f"\n Проверка:")
print(f"   Размер train.jsonl.gz: {TRAIN_GZ.stat().st_size / 1024 / 1024:.1f} MB")
print(f"   Размер valid.jsonl.gz: {VALID_GZ.stat().st_size / 1024 / 1024:.1f} MB")

print("\nТеперь можно запускать обучение AudioCraft:")
print("cd audiocraft")
print("dora run solver=musicgen/musicgen_finetune \\")
print("  model/lm/model_scale=small \\")
print("  dataset.batch_size=4")