"""
Модуль для обработки аудио и определения эмоций животных по звукам
Совместим с Python 3.13.1
"""
import numpy as np
import librosa
import os
from typing import Tuple, Optional

class AudioEmotionDetector:
    """Класс для определения эмоций животных по аудио"""
    
    # Маппинг эмоций (совпадает с моделью изображений)
    EMOTION_MAP = {
        0: 'angry',    # Злой/Агрессивный
        1: 'happy',    # Счастливый
        2: 'relaxed',  # Расслабленный
        3: 'sad'       # Грустный
    }
    
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 3  # секунды для анализа
        
    def extract_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Извлекает признаки из аудио файла
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Массив признаков или None в случае ошибки
        """
        try:
            # Загружаем аудио
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Проверяем, что аудио не пустое
            if len(y) == 0:
                print("Предупреждение: аудио файл пуст")
                return None
            
            # Извлекаем различные признаки
            features = []
            
            # 1. Zero Crossing Rate (частота пересечения нуля) - показывает агрессивность
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            features.append(zcr)
            
            # 2. Spectral Centroid (спектральный центроид) - яркость звука
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.append(np.mean(spectral_centroids))
            
            # 3. Spectral Rolloff (спектральный спад)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.append(np.mean(rolloff))
            
            # 4. MFCC (Mel-frequency cepstral coefficients) - основные признаки
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            
            # 5. Chroma features (хроматические признаки)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # 6. Tempo (темп) - если доступен
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))
            except Exception:
                features.append(0.0)
            
            # 7. RMS Energy (энергия)
            rms = librosa.feature.rms(y=y)[0]
            features.append(np.mean(rms))
            
            return np.array(features)
            
        except librosa.util.exceptions.NoBackendError:
            print("Ошибка: не найден бэкенд для обработки аудио. Установите soundfile или ffmpeg")
            return None
        except Exception as e:
            print(f"Ошибка при обработке аудио: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_emotion(self, audio_path: str) -> Tuple[int, float, str]:
        """
        Определяет эмоцию по аудио файлу
        
        Args:
            audio_path: Путь к аудио файлу
            
        Returns:
            Tuple (индекс эмоции, уверенность, название эмоции)
        """
        if not os.path.exists(audio_path):
            print(f"Ошибка: файл {audio_path} не найден")
            return 2, 0.3, 'relaxed'
        
        features = self.extract_features(audio_path)
        
        if features is None or len(features) == 0:
            return 2, 0.5, 'relaxed'  # По умолчанию расслабленный
        
        # Простая эвристическая модель на основе признаков
        # В реальном проекте здесь должна быть обученная модель
        
        # Нормализуем признаки
        features = features / (np.abs(features).max() + 1e-8)
        
        # Эвристики для определения эмоций:
        zcr = features[0]  # Zero Crossing Rate
        energy = features[-1]  # RMS Energy
        spectral_centroid = features[1]
        
        # Высокий ZCR и высокая энергия -> злой/агрессивный
        if zcr > 0.15 and energy > 0.3:
            emotion_idx = 0  # angry
            confidence = min(0.85, zcr * 2 + energy)
        
        # Средний ZCR, высокая энергия, высокий спектральный центроид -> счастливый
        elif energy > 0.25 and spectral_centroid > 0.4:
            emotion_idx = 1  # happy
            confidence = min(0.9, energy * 2 + spectral_centroid * 0.5)
        
        # Низкая энергия, низкий ZCR -> грустный
        elif energy < 0.15 and zcr < 0.1:
            emotion_idx = 3  # sad
            confidence = min(0.8, (0.15 - energy) * 3 + (0.1 - zcr) * 2)
        
        # Остальные случаи -> расслабленный
        else:
            emotion_idx = 2  # relaxed
            confidence = 0.7
        
        emotion_name = self.EMOTION_MAP[emotion_idx]
        return int(emotion_idx), float(confidence), str(emotion_name)

