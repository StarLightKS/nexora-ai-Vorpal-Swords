# Установка для Python 3.13.1

## Требования

- **Python 3.13.1** или выше
- pip (менеджер пакетов Python)

## Пошаговая установка

### 1. Проверьте версию Python

```bash
python --version
```

Должно быть: `Python 3.13.1` или выше

### 2. Создайте виртуальное окружение (рекомендуется)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Обновите pip

```bash
python -m pip install --upgrade pip
```

### 4. Установите зависимости

```bash
pip install -r requirements.txt
```

### 5. Дополнительные зависимости для аудио (опционально)

Для работы с аудио файлами может потребоваться:

**Windows:**
- Установите [FFmpeg](https://ffmpeg.org/download.html) или используйте `pip install soundfile`

**Linux:**
```bash
sudo apt-get install ffmpeg libsndfile1
```

**MacOS:**
```bash
brew install ffmpeg libsndfile
```

### 6. Запустите приложение

```bash
python app.py
```

Приложение будет доступно по адресу: **http://127.0.0.1:5000**

## Решение проблем

### Проблема: TensorFlow не устанавливается

Если TensorFlow не устанавливается для Python 3.13.1, попробуйте:

```bash
pip install tensorflow --upgrade
```

Или используйте предварительную версию:
```bash
pip install tf-nightly
```

### Проблема: librosa не работает

Убедитесь, что установлены все зависимости:
```bash
pip install librosa soundfile resampy numba
```

### Проблема: Модель не загружается

Убедитесь, что файл `pet_emotion.h5` находится в корневой директории проекта.

## Проверка установки

Запустите проверку:

```bash
python -c "import tensorflow as tf; import flask; import librosa; print('Все зависимости установлены успешно!')"
```

Если команда выполнилась без ошибок, установка прошла успешно!

