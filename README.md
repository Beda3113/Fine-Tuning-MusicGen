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

[**ЯНДЕКС.ДИСКА**](https://disk.yandex.ru/client/disk/УЧЕБА%20/data/musiccaps_complete)

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


## ЧАСТИ 1.1: Сбор данных MusicCaps

```
python scripts/part1.2_enrich_metadata.py
```

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

[**ДАТАСЕТ ЯНДЕКС.ДИСКА**](https://disk.yandex.ru/client/disk/УЧЕБА%20/data/musiccaps_complete/audio)

## Часть 1.2: Обогащение метаданных
```
ollama serve
```
```
ollama pull llama3.2
```
```
part1.2_enrich_metadata.py
```

Выполненная работа
 Разработка скрипта обогащения
 Создан скрипт part1.2_enrich_metadata.py, который выполняет следующие функции:

- Загрузка исходных описаний из папки metadata/
- Отправка запросов к локальной Llama 3 через Ollama с использованием V100 GPU
- Извлечение структурированных данных в соответствии с требуемой JSON-схемой
- Валидация и дополнение отсутствующих полей
- Сохранение результатов в двух местах:
* В папке enriched/ для организации
* Рядом с WAV файлами в audio/ (согласно требованию)

[**metadata ЯНДЕКС.ДИСКА**](https://disk.yandex.ru/client/disk/УЧЕБА%20/data/musiccaps_complete/audio/metadata)


## Часть 1.3: Модификация AudioCraft


1. Клон репозитория AudioCraft
```
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
```
2. Модификация файла music_dataset.py

Путь к файлу: audiocraft/audiocraft/data/music_dataset.py

```
@dataclass
class MusicInfo(AudioInfo):
    # ... существующие поля ...
    
    # НОВЫЕ ПОЛЯ ИЗ JSON-СХЕМЫ
    general_mood: tp.Optional[str] = None          # общее настроение
    genre_tags: tp.Optional[list] = None           # теги жанров (список)
    lead_instrument: tp.Optional[str] = None       # основной инструмент
    accompaniment: tp.Optional[str] = None         # аккомпанемент
    tempo_and_rhythm: tp.Optional[str] = None      # темп и ритм
    vocal_presence: tp.Optional[str] = None        # вокал
    production_quality: tp.Optional[str] = None    # качество продакшна
```

Обновленный метод attribute_getter:
```
@staticmethod
def attribute_getter(attribute):
    if attribute == 'bpm':
        preprocess_func = get_bpm
    elif attribute == 'key':
        preprocess_func = get_musical_key
    elif attribute in ['moods', 'keywords', 'genre_tags']:  # ДОБАВЛЕНО genre_tags
        preprocess_func = get_keyword_list
    elif attribute in ['genre', 'name', 'instrument', 'lead_instrument']:  # ДОБАВЛЕНО lead_instrument
        preprocess_func = get_keyword
    elif attribute in ['title', 'artist', 'description', 'general_mood', 
                      'accompaniment', 'tempo_and_rhythm', 'vocal_presence', 
                      'production_quality']:  # ДОБАВЛЕНЫ все новые текстовые поля
        preprocess_func = get_string
    else:
        preprocess_func = None
    return preprocess_func
```
Обновленный метод to_condition_attributes:

```
def to_condition_attributes(self) -> ConditioningAttributes:
    out = ConditioningAttributes()
    for _field in fields(self):
        key, value = _field.name, getattr(self, _field.name)
        if key == 'self_wav':
            out.wav[key] = value
        elif key == 'joint_embed':
            for embed_attribute, embed_cond in value.items():
                out.joint_embed[embed_attribute] = embed_cond
        else:
            if isinstance(value, list):
                value = ' '.join(value)
            out.text[key] = value  # ВСЕ ПОПАДАЮТ В ТЕКСТОВЫЕ УСЛОВИЯ
    return out
```

Модифицированный AudioCraft готов к использованию структурированных данных для обучения


## Части 1.4: Настройка конфигов и запуск обучения

1. Создание манифестов train/valid
2. Создание конфигурационных файлов
 2.1 Конфиг датасета audiocraft/config/dset/audio/musiccaps.yaml
```
   # @package __global__

datasource:
  max_sample_rate: 32000
  max_channels: 1
  
  train: C:/Users/user/Desktop/последняя домашка DL/data/musiccaps_complete/train.jsonl.gz
  valid: C:/Users/user/Desktop/последняя домашка DL/data/musiccaps_complete/valid.jsonl.gz
  evaluate: C:/Users/user/Desktop/последняя домашка DL/data/musiccaps_complete/valid.jsonl.gz
  generate: C:/Users/user/Desktop/последняя домашка DL/data/musiccaps_complete/valid.jsonl.gz
  ```
2.2 Конфиг обучения audiocraft/config/solver/musicgen/musicgen_finetune.yaml

```
# @package __global__

defaults:
  - musicgen/default
  - /model: lm/musicgen_lm
  - override /dset: audio/musiccaps
  - _self_

autocast: true
autocast_dtype: float16

# Используем предобученную MusicGen-small
compression_model_checkpoint: //reference/facebook/musicgen-small

channels: 1
sample_rate: 32000

deadlock:
  use: true

dataset:
  batch_size: 4  # Для V100 16GB
  num_workers: 4
  segment_duration: 10
  min_segment_ratio: 1.0
  sample_on_weight: false
  sample_on_duration: false
  
  train:
    # Параметры для работы с новыми полями и CFG
    merge_text_p: 0.25    # 25% - объединение всех полей
    drop_desc_p: 0.5      # 50% - удаление description
    drop_other_p: 0.5     # 50% - удаление других полей

optim:
  epochs: 10
  optimizer: adamw
  lr: 1e-4
  ema:
    use: true
    updates: 10
    device: cuda

logging:
  log_tensorboard: true
  log_wandb: false

schedule:
  lr_scheduler: inverse_sqrt
  inverse_sqrt:
    warmup: 1500
    warmup_init_lr: 0.0

checkpoint:
  save_every: 5
  keep_last: 3
```

