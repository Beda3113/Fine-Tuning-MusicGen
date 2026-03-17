"""
MusicCaps Metadata Enrichment with Llama 3 on V100
Часть 1.2 ДЗ4
"""

import os
import json
import time
from pathlib import Path
import logging
from tqdm import tqdm
import subprocess
import shutil
import sys
from datetime import datetime

log_filename = f'enrichment_{datetime.now().strftime("%Y%m%d_%H%M")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MusicCapsEnricher:
    def __init__(self, base_path="data/musiccaps_complete", model="llama3.2"):
        self.base_path = Path(base_path)
        self.audio_dir = self.base_path / "audio"
        self.metadata_dir = self.base_path / "metadata"
        self.output_dir = self.base_path / "enriched"
        self.model = model
        
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'moved': 0
        }
        
        self.system_prompt = self._get_system_prompt()
        
        logging.info(f"Инициализация: {self.base_path}")
        logging.info(f"Модель: {self.model}")
    
    def _get_system_prompt(self):
        return """You are an expert music analyst. Parse a raw text description of a 10-second music clip and extract specific information into a JSON object.

The description comes from the MusicCaps dataset and describes a short music fragment.

You must output ONLY a valid JSON object with exactly these fields:
{
  "description": "A concise summary of the music (1-2 sentences)",
  "general_mood": "The overall mood or emotion (e.g., happy, sad, energetic, calm)",
  "genre_tags": ["list", "of", "relevant", "genres", "or", "styles"],
  "lead_instrument": "The main instrument or voice",
  "accompaniment": "Supporting instruments or sounds",
  "tempo_and_rhythm": "Description of tempo and rhythmic feel",
  "vocal_presence": "none/solo/choir/rap/etc.",
  "production_quality": "Description of recording quality and production style"
}

Guidelines:
- Use information ONLY from the provided description. Do not invent details.
- If a field cannot be inferred from the text, use an empty string or empty list.
- Keep genre_tags as a list of strings, even if only one genre.
- Be specific but concise.
- Output valid JSON only, no additional text or explanations."""
    
    def check_environment(self):
        logging.info("ПРОВЕРКА ОКРУЖЕНИЯ")
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if 'V100' in result.stdout:
                os.environ['CUDA_VISIBLE_DEVICES'] = '1'
                logging.info("V100 найдена")
            else:
                logging.warning("V100 не найдена")
        except:
            logging.warning("GPU не найдена")
        
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, shell=True)
            logging.info(f"Ollama: {result.stdout.strip()}")
            
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, shell=True)
            if self.model not in result.stdout:
                logging.warning(f"Модель {self.model} не найдена")
                print(f"Скачайте: ollama pull {self.model}")
                return False
        except:
            logging.error("Ollama не установлена")
            print("Установите Ollama с https://ollama.ai/download/windows")
            return False
        
        return True
    
    def organize_files(self):
        logging.info("ОРГАНИЗАЦИЯ ФАЙЛОВ")
        
        txt_files = list(Path(".").glob("**/*.txt"))
        logging.info(f"Найдено .txt файлов: {len(txt_files)}")
        
        moved = 0
        for txt_file in txt_files:
            if any(x in str(txt_file).lower() for x in ['cookies', 'log', 'debug', 'license']):
                continue
            
            dest_path = self.metadata_dir / txt_file.name
            if txt_file.parent != self.metadata_dir:
                try:
                    shutil.copy2(txt_file, dest_path)
                    moved += 1
                except Exception as e:
                    logging.error(f"Ошибка копирования {txt_file.name}: {e}")
        
        self.stats['moved'] = moved
        logging.info(f"Перемещено в metadata: {moved} файлов")
        return moved
    
    def check_files(self):
        logging.info("ПРОВЕРКА ФАЙЛОВ")
        
        wav_files = list(self.audio_dir.glob("*.wav"))
        txt_files = list(self.metadata_dir.glob("*.txt"))
        
        wav_ids = {f.stem for f in wav_files}
        txt_ids = {f.stem for f in txt_files}
        common_ids = wav_ids & txt_ids
        
        logging.info(f"Аудиофайлов: {len(wav_files)}")
        logging.info(f"Текстовых файлов: {len(txt_files)}")
        logging.info(f"Файлов с обоими: {len(common_ids)}")
        
        if len(common_ids) == 0:
            logging.error("Нет пар аудио/текст для обработки")
            return None
        
        return [f for f in txt_files if f.stem in common_ids]
    
    def call_llama(self, caption):
        for attempt in range(3):
            try:
                full_prompt = f"""{self.system_prompt}

Input description: {caption}

Output JSON:"""
                
                full_prompt = full_prompt.replace('"', '\\"').replace('\n', ' ')
                cmd = f'ollama run {self.model} "{full_prompt}"'
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    shell=True,
                    encoding='utf-8',
                    timeout=45,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}
                )
                
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    logging.warning(f"Попытка {attempt+1}: {result.stderr[:100]}")
                    time.sleep(2)
                    
            except subprocess.TimeoutExpired:
                logging.warning(f"Попытка {attempt+1}: таймаут")
                time.sleep(2)
            except Exception as e:
                logging.warning(f"Попытка {attempt+1}: {e}")
                time.sleep(2)
        
        return None
    
    def parse_json_response(self, response_text, ytid):
        try:
            response_text = response_text.strip()
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end > start:
                metadata = json.loads(response_text[start:end])
            else:
                metadata = json.loads(response_text)
            
            required_fields = [
                'description', 'general_mood', 'genre_tags',
                'lead_instrument', 'accompaniment', 'tempo_and_rhythm',
                'vocal_presence', 'production_quality'
            ]
            
            for field in required_fields:
                if field not in metadata:
                    if field == 'genre_tags':
                        metadata[field] = []
                    else:
                        metadata[field] = ""
                    logging.debug(f"{ytid}: добавлено поле '{field}'")
            
            return metadata
            
        except json.JSONDecodeError as e:
            logging.error(f"{ytid}: ошибка парсинга JSON - {e}")
            debug_file = self.base_path / f"debug_{ytid}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            return None
    
    def process_file(self, txt_file):
        ytid = txt_file.stem
        json_output = self.output_dir / f"{ytid}.json"
        json_next_to_wav = self.audio_dir / f"{ytid}.json"
        
        if json_output.exists():
            return "skipped"
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except Exception as e:
            logging.error(f"{ytid}: ошибка чтения файла - {e}")
            return "failed"
        
        if not caption:
            logging.warning(f"{ytid}: пустой файл описания")
            return "failed"
        
        response = self.call_llama(caption)
        if not response:
            return "failed"
        
        metadata = self.parse_json_response(response, ytid)
        if not metadata:
            return "failed"
        
        metadata['ytid'] = ytid
        metadata['original_caption'] = caption
        metadata['model'] = self.model
        metadata['enriched_date'] = datetime.now().isoformat()
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        shutil.copy2(json_output, json_next_to_wav)
        
        return "success"
    
    def process_all(self, limit=None):
        logging.info("НАЧАЛО ОБРАБОТКИ")
        
        if not self.check_environment():
            return
        
        self.organize_files()
        files_to_process = self.check_files()
        
        if not files_to_process:
            return
        
        if limit:
            files_to_process = files_to_process[:limit]
        
        self.stats['total'] = len(files_to_process)
        logging.info(f"Будет обработано файлов: {self.stats['total']}")
        
        if files_to_process:
            logging.info("Тестирование первого файла...")
            test_status = self.process_file(files_to_process[0])
            logging.info(f"Результат теста: {test_status}")
            
            if test_status == 'failed':
                print("Тест не пройден. Проверьте:")
                print("1. Работает ли Ollama? (ollama list)")
                print(f"2. Загружена ли модель? (ollama pull {self.model})")
                response = input("Продолжить обработку всех файлов? (y/n): ")
                if response.lower() != 'y':
                    return
        
        with tqdm(total=self.stats['total'], desc="Обработка") as pbar:
            for txt_file in files_to_process:
                status = self.process_file(txt_file)
                
                if status == 'success':
                    self.stats['success'] += 1
                elif status == 'skipped':
                    self.stats['skipped'] += 1
                else:
                    self.stats['failed'] += 1
                
                pbar.update(1)
                pbar.set_postfix(success=self.stats['success'], failed=self.stats['failed'])
                time.sleep(1)
        
        self.create_manifest()
        self.print_report()
    
    def create_manifest(self):
        logging.info("СОЗДАНИЕ МАНИФЕСТА")
        
        json_files = list(self.output_dir.glob("*.json"))
        manifest_file = self.base_path / "manifest.jsonl"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    data['audio_file'] = f"audio/{data['ytid']}.wav"
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logging.info(f"Манифест создан: {manifest_file}")
        logging.info(f"Записей в манифесте: {len(json_files)}")
    
    def print_report(self):
        print(f"\nИТОГОВЫЙ ОТЧЕТ ОБОГАЩЕНИЯ МЕТАДАННЫХ")
        print(f"Базовая папка: {self.base_path}")
        print(f"Исходных .txt файлов: {self.stats['moved']}")
        print(f"Всего для обработки: {self.stats['total']}")
        print(f"Успешно обработано: {self.stats['success']}")
        print(f"Пропущено (уже были): {self.stats['skipped']}")
        print(f"Ошибок: {self.stats['failed']}")
        
        json_files = list(self.output_dir.glob("*.json"))
        json_in_audio = list(self.audio_dir.glob("*.json"))
        
        print(f"\nРЕЗУЛЬТИРУЮЩИЕ ФАЙЛЫ:")
        print(f"JSON в enriched/: {len(json_files)}")
        print(f"JSON рядом с .wav: {len(json_in_audio)}")
        print(f"Манифест: {self.base_path}/manifest.jsonl")
        
        if json_files:
            print(f"\nПРИМЕР ПЕРВОГО JSON:")
            with open(json_files[0], 'r', encoding='utf-8') as f:
                example = json.load(f)
                print(json.dumps(example, indent=2, ensure_ascii=False)[:500] + "...")
        
        print(f"\nЧАСТЬ 1.2 ЗАВЕРШЕНА")
    
    def verify_schema(self):
        logging.info("ПРОВЕРКА СХЕМЫ JSON")
        
        json_files = list(self.output_dir.glob("*.json"))
        required_fields = [
            'description', 'general_mood', 'genre_tags',
            'lead_instrument', 'accompaniment', 'tempo_and_rhythm',
            'vocal_presence', 'production_quality', 'ytid'
        ]
        
        valid = 0
        invalid = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if all(field in data for field in required_fields):
                    valid += 1
                else:
                    missing = [f for f in required_fields if f not in data]
                    logging.warning(f"{json_file.name}: отсутствуют поля {missing}")
                    invalid += 1
                    
            except Exception as e:
                logging.error(f"{json_file.name}: ошибка - {e}")
                invalid += 1
        
        logging.info(f"Валидных JSON: {valid}")
        logging.info(f"Проблемных JSON: {invalid}")
        
        return valid, invalid

def main():
    print("ЧАСТЬ 1.2: ОБОГАЩЕНИЕ МЕТАДАННЫХ MUSICCAPS")
    
    default_path = "data/musiccaps_complete"
    path_input = input(f"Путь к данным (Enter для '{default_path}'): ").strip()
    base_path = path_input if path_input else default_path
    
    print("\nДоступные модели Ollama:")
    print("1. llama3.2 - быстрая, хорошее качество")
    print("2. llama3 - медленнее, лучше качество")
    print("3. phi3:mini - самая быстрая")
    print("4. mistral - альтернатива")
    
    choice = input("Выберите модель (1/2/3/4) [1]: ").strip() or "1"
    
    models = {
        '1': 'llama3.2',
        '2': 'llama3',
        '3': 'phi3:mini',
        '4': 'mistral'
    }
    model = models.get(choice, 'llama3.2')
    
    limit_input = input("Сколько файлов обработать (Enter для всех, 5 для теста): ").strip()
    limit = int(limit_input) if limit_input else None
    
    print(f"\nЗАПУСК С ПАРАМЕТРАМИ:")
    print(f"Путь к данным: {base_path}")
    print(f"Модель: {model}")
    print(f"Лимит: {limit if limit else 'все файлы'}")
    
    enricher = MusicCapsEnricher(base_path=base_path, model=model)
    enricher.process_all(limit=limit)
    
    if enricher.stats['success'] > 0:
        print("\nДополнительная проверка JSON...")
        enricher.verify_schema()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        print(f"Ошибка: {e}")
        sys.exit(1)
