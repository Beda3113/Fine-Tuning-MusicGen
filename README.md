# Домашнее задание 4: Fine-tuning MusicGen


```
│
├── ДЗ4_отчет.md
├── README.md
│
├── data\
│   └── musiccaps_complete\
│       ├── audio\
│       │   ├── E4To9BC2jx8.wav
│       │   ├── ZkGQhIbEDrs.wav
│       │   └── ...
│       │
│       ├── metadata\
│       │   ├── E4To9BC2jx8.txt
│       │   ├── ZkGQhIbEDrs.txt
│       │   └── ...
│       │
│       ├── enriched\
│       │   ├── E4To9BC2jx8.json
│       │   ├── ZkGQhIbEDrs.json
│       │   └── ...
│       │
│       ├── manifest.jsonl
│       └── dataset_stats.json
│
├── scripts\
│   ├── part1.1_download.py
│   ├── part1.2_enrich_metadata.py
│   ├── part1.3_test_audiocraft.py
│   ├── part1.4_train.py
│   └── part2_generate.py
│
├── configs\
│   ├── train_musiccaps.yaml
│   └── generation.yaml
│
├── output\
│   ├── generated\
│   │   ├── prompt_1.wav
│   │   ├── prompt_2.wav
│   │   ├── prompt_3.wav
│   │   ├── prompt_4.wav
│   │   └── prompt_5.wav
│   └── logs\
│       ├── training.log
│       └── generation.log
│
├── audiocraft\
│   ├── audiocraft\
│   │   ├── data\
│   │   │   └── music_dataset.py
│   │   └── ...
│   ├── config\
│   │   └── ...
│   └── setup.py
│
├── model_weights\
│   ├── musicgen_finetuned\
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── ...
│   └── model_link.txt
│
├── tests\
│   ├── test_json_schema.py
│   └── test_audio_duration.py
│
├── enrichment.log
├── training.log
├── setup_v100.ps1
└── requirements.txt
```


## Структура проекта
- `data/musiccaps_complete/` - датасет MusicCaps
- `scripts/` - скрипты для всех этапов
- `audiocraft/` - модифицированный AudioCraft
- `output/generated/` - сгенерированные треки
- `model_weights/` - веса обученной модели




## Установка
```bash
# 1. Клонировать репозиторий
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .

# 2. Установить зависимости
pip install -r requirements.txt
```


##ЧАСТИ 1.1: Сбор данных MusicCaps

Что было сделано:
1. Разработан скрипт скачивания part1.1_download.py

 - Загрузка метаданных с HuggingFace
 - Использование yt-dlp + ffmpeg для вырезания 10-секундных фрагментов
 - Параллельная загрузка с контролем потоков

2. Полученные данные:

```
data/musiccaps_complete/
├── audio/          # 3132 WAV файла (10 сек, ~1 MB каждый)
└── metadata/       # 3132 TXT файла с описаниями
```
Характеристики датасета:


## Характеристики полученного датасета

| Параметр | Значение |
|----------|----------|
| Всего файлов в датасете | 5521 |
| Успешно скачано | 3132 |
| Недействительные ссылки | 2311 |
| Ошибки при скачивании | 78 |
| Процент успеха | 57% |
| Общий объем данных | 3 GB |
| Средний размер файла | 1 MB |
| Длительность | 10 секунд |
| Формат | WAV, моно |

## Проблемы и решения

| Проблема | Решение | Результат |
|----------|----------|-----------|
| Блокировка YouTube | Использование cookies + паузы между запросами | Удалось скачать 3132 файла |
| Отсутствие cookies | Экспорт через расширение браузера "Get cookies.txt" | Успешная авторизация |
| Недействительные ссылки | Пропуск видео, помеченных как удаленные/приватные | 2311 файлов пропущено |
| Ошибки соединения | Повторные попытки (3 retries) | Только 78 ошибок |



Часть 1.2: Обогащение метаданных
```
ollama serve
```
```
ollama pull llama3.2
```
```
part1.2_enrich_metadata.py
```
