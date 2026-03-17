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
