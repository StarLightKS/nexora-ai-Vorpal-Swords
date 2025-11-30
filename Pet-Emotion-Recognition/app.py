from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
from werkzeug.utils import secure_filename
from audio_processor import AudioEmotionDetector
from ai_advisor import AIAdvisor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Кэширование модели для улучшения производительности
_model_cache = None
_audio_detector = None
_ai_advisor = None

def get_image_model():
    """Загружает модель изображений с кэшированием"""
    global _model_cache
    if _model_cache is None:
        try:
            model_path = "pet_emotion.h5"
            if not os.path.exists(model_path):
                print(f"Предупреждение: файл модели {model_path} не найден")
                return None
            # Используем compile=False для ускорения загрузки (если не нужна компиляция)
            _model_cache = load_model(model_path, compile=False)
            print("Модель изображений загружена успешно")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _model_cache

def get_audio_detector():
    """Получает детектор аудио эмоций"""
    global _audio_detector
    if _audio_detector is None:
        _audio_detector = AudioEmotionDetector()
    return _audio_detector

def get_ai_advisor():
    """Получает AI советника"""
    global _ai_advisor
    if _ai_advisor is None:
        _ai_advisor = AIAdvisor()
    return _ai_advisor

# Маппинг эмоций
EMOTION_MAP = {
    0: 'angry',
    1: 'happy',
    2: 'relaxed',
    3: 'sad'
}

EMOTION_NAMES_RU = {
    'angry': 'Злой',
    'happy': 'Счастливый',
    'relaxed': 'Расслабленный',
    'sad': 'Грустный'
}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/pet-emotion", methods=['GET', 'POST'])
def PetEmotionPage():
    return render_template('index.html')

def allowed_file(filename, allowed_extensions):
    """Проверяет разрешенные расширения файлов"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route("/pet-emotion-predict", methods=['POST', 'GET'])
def pet_emotion_predictPage():
    pred = None
    emotion_name = None
    confidence = 0.0
    input_type = None
    advice_data = None
    
    if request.method == 'POST':
        try:
            # Обработка изображения
            if 'image' in request.files and request.files['image'].filename:
                file = request.files['image']
                if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'gif', 'bmp'}):
                    input_type = 'image'
                    img = Image.open(file.stream)
                    
                    # Конвертируем RGBA в RGB если необходимо
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    img = img.resize((224, 224))
                    x = np.asarray(img)
                    
                    # Проверяем размерность
                    if len(x.shape) == 2:  # Grayscale
                        x = np.stack([x, x, x], axis=-1)
                    elif x.shape[2] == 4:  # RGBA
                        x = x[:, :, :3]
                    
                    x = np.expand_dims(x, axis=0)
                    x = x / 255.0

                    # Загружаем модель
                    model = get_image_model()
                    if model is None:
                        raise Exception("Модель изображений не найдена. Убедитесь, что файл pet_emotion.h5 находится в корневой директории проекта.")

                    # Делаем предсказание
                    predictions = model.predict(x, verbose=0)
                    pred = int(np.argmax(predictions))
                    confidence = float(np.max(predictions))
                    emotion_name = EMOTION_MAP.get(pred, 'unknown')
                    
            # Обработка аудио
            elif 'audio' in request.files and request.files['audio'].filename:
                file = request.files['audio']
                if file and allowed_file(file.filename, {'wav', 'mp3', 'ogg', 'm4a', 'flac'}):
                    input_type = 'audio'
                    
                    # Сохраняем временный файл
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(temp_path)
                    
                    try:
                        # Определяем эмоцию по аудио
                        audio_detector = get_audio_detector()
                        pred, confidence, emotion_name = audio_detector.predict_emotion(temp_path)
                    finally:
                        # Удаляем временный файл
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            # Генерируем совет ИИ
            if pred is not None and emotion_name:
                advisor = get_ai_advisor()
                advice_data = advisor.get_advice(pred, emotion_name, confidence)
                advice_data['emotion_name_ru'] = EMOTION_NAMES_RU.get(emotion_name, emotion_name)
                
        except Exception as e:
            print(f"Ошибка при распознавании: {e}")
            import traceback
            traceback.print_exc()
            message = f"Ошибка при распознавании: {str(e)}. Пожалуйста, попробуйте снова."
            return render_template('index.html', message=message)
    
    return render_template('predict.html', 
                         pred=pred, 
                         emotion_name=emotion_name,
                         emotion_name_ru=EMOTION_NAMES_RU.get(emotion_name, emotion_name) if emotion_name else None,
                         confidence=confidence,
                         input_type=input_type,
                         advice_data=advice_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
