import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
from PIL import Image, ImageDraw, ImageFont
import os
from flask import Flask, Response, render_template, request, jsonify
import threading
import queue
import base64

app = Flask(__name__)

# Глобальные переменные для обмена данными между потоками
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
current_frame = None
processing_active = True

class SquatTrainer:
    def __init__(self, model_name='yolov8n-pose.pt', conf_threshold=0.7):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Key points for squat analysis
        self.body_points = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        # Current state
        self.squat_depth = 0
        self.feedback_messages = []
        self.knee_feedback = []
        self.rep_count = 0
        self.is_squatting = False
        self.min_knee_angle = 180
        self.view_angle = "unknown"

        # Colors for feedback
        self.colors = {
            'good': (0, 255, 0),
            'warning': (0, 255, 255),
            'bad': (0, 0, 255),
            'info': (255, 255, 255)
        }

        # Загрузка шрифта
        self.font_path = self.get_font_path()
        self.pil_font_medium = ImageFont.truetype(self.font_path, 20) if self.font_path else None
        self.pil_font_small = ImageFont.truetype(self.font_path, 16) if self.font_path else None

    def get_font_path(self):
        """Поиск подходящего шрифта для русского текста"""
        possible_fonts = [
            'arial.ttf', 'arialbd.ttf', 'DejaVuSans.ttf',
            'LiberationSans-Regular.ttf', 'Roboto-Regular.ttf'
        ]

        font_paths = [
            '/usr/share/fonts/truetype/freefont/',
            '/usr/share/fonts/truetype/dejavu/',
            '/usr/share/fonts/truetype/liberation/',
            'C:/Windows/Fonts/',
            '/Library/Fonts/'
        ]

        for font_path in font_paths:
            for font_name in possible_fonts:
                full_path = os.path.join(font_path, font_name)
                if os.path.exists(full_path):
                    return full_path
        return None

    def put_ru_text(self, image, text, position, font_size='medium', color=(255, 255, 255)):
        """Функция для отображения русского текста на изображении"""
        try:
            if font_size == 'medium' and self.pil_font_medium:
                font = self.pil_font_medium
            elif font_size == 'small' and self.pil_font_small:
                font = self.pil_font_small
            else:
                cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                return image

            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return image

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (b - vertex)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def determine_view_angle(self, keypoints):
        """Determine if the view is front, side, or unknown"""
        try:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            hip_width = abs(left_hip[0] - right_hip[0])

            if shoulder_width > 100 and hip_width > 80:
                return "фронтальный"
            elif shoulder_width < 60 and hip_width < 50:
                return "профиль"
            else:
                return "неопределен"
        except:
            return "неопределен"

    def analyze_knee_position(self, keypoints):
        """Analyze knee position relative to feet and hips"""
        knee_feedback = []
        try:
            left_hip = [keypoints[11][0], keypoints[11][1]]
            right_hip = [keypoints[12][0], keypoints[12][1]]
            left_knee = [keypoints[13][0], keypoints[13][1]]
            right_knee = [keypoints[14][0], keypoints[14][1]]
            left_ankle = [keypoints[15][0], keypoints[15][1]]
            right_ankle = [keypoints[16][0], keypoints[16][1]]

            # Check knee symmetry
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            knee_diff = abs(left_knee_angle - right_knee_angle)

            if knee_diff > 15:
                knee_feedback.append("Колени несимметричны!")
            else:
                knee_feedback.append("Симметрия: Хорошо")

            # Check knee stability
            left_knee_over_toe = abs(left_knee[0] - left_ankle[0]) < 100
            right_knee_over_toe = abs(right_knee[0] - right_ankle[0]) < 100

            if left_knee_over_toe and right_knee_over_toe:
                knee_feedback.append("Положение: Хорошо")
            else:
                knee_feedback.append("Колени выходят за стопы!")

        except Exception as e:
            knee_feedback.append("Анализ коленей: Недостаточно данных")
        return knee_feedback

    def analyze_squat_form(self, keypoints):
        """Analyze squat form and provide feedback"""
        feedback = []
        warnings = []
        try:
            self.view_angle = self.determine_view_angle(keypoints)

            # Get keypoint coordinates
            left_hip = [keypoints[11][0], keypoints[11][1]]
            right_hip = [keypoints[12][0], keypoints[12][1]]
            left_knee = [keypoints[13][0], keypoints[13][1]]
            right_knee = [keypoints[14][0], keypoints[14][1]]
            left_ankle = [keypoints[15][0], keypoints[15][1]]
            right_ankle = [keypoints[16][0], keypoints[16][1]]
            left_shoulder = [keypoints[5][0], keypoints[5][1]]
            right_shoulder = [keypoints[6][0], keypoints[6][1]]

            # Calculate angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2

            # Analyze knee position
            self.knee_feedback = self.analyze_knee_position(keypoints)

            # Update minimum knee angle for depth tracking
            if knee_angle < self.min_knee_angle:
                self.min_knee_angle = knee_angle

            # Squat depth analysis
            squat_percentage = max(0, min(100, (180 - knee_angle) / 90 * 100))
            self.squat_depth = squat_percentage

            # Knee analysis
            if knee_angle < 60:
                warnings.append("Слишком глубоко!")
            elif knee_angle > 120:
                feedback.append("Опуститесь глубже")
            elif 80 <= knee_angle <= 100:
                feedback.append("Идеальная глубина")
            else:
                feedback.append("Хорошая глубина")

            # Repetition tracking
            if knee_angle < 100 and not self.is_squatting:
                self.is_squatting = True
            elif knee_angle > 160 and self.is_squatting:
                self.is_squatting = False
                self.rep_count += 1
                self.min_knee_angle = 180
                feedback.append(f"Повторение {self.rep_count}!")

            feedback.append(f"Ракурс: {self.view_angle}")

            return feedback, warnings, {
                'knee_angle': knee_angle,
                'squat_depth': squat_percentage
            }

        except Exception as e:
            return ["Не все точки тела видны"], [], {}

    def draw_body_points(self, result, frame):
        """Draw body points with additional information"""
        if result.keypoints is None:
            return frame

        annotated_frame = frame.copy()
        keypoints = result.keypoints.data.cpu().numpy()

        for person_kpts in keypoints:
            # Analyze pose
            feedback, warnings, angles = self.analyze_squat_form(person_kpts)
            self.feedback_messages = feedback + warnings

            # Draw body points
            for point_id in self.body_points:
                if point_id < len(person_kpts) and person_kpts[point_id][2] > 0.3:
                    x, y = int(person_kpts[point_id][0]), int(person_kpts[point_id][1])
                    
                    if point_id in [13, 14]:  # Knees
                        color = self.colors['good'] if angles.get('knee_angle', 0) > 80 and angles.get('knee_angle', 0) < 100 else self.colors['warning']
                    elif point_id in [11, 12]:  # Hips
                        color = self.colors['good']
                    else:
                        color = self.colors['info']

                    cv2.circle(annotated_frame, (x, y), 6, color, -1)
                    cv2.circle(annotated_frame, (x, y), 8, (255, 255, 255), 2)

        return annotated_frame

    def add_fitness_info(self, frame, result):
        """Add fitness information to frame"""
        h, w = frame.shape[:2]

        # Info panel
        panel_width = 400
        panel_height = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Info text
        y_offset = 40
        line_height = 25

        frame = self.put_ru_text(frame, f"Повторения: {self.rep_count}", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        frame = self.put_ru_text(frame, f"Глубина: {self.squat_depth:.1f}%", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height

        frame = self.put_ru_text(frame, f"Ракурс: {self.view_angle}", (20, y_offset),
                                 font_size='medium', color=self.colors['info'])
        y_offset += line_height + 10

        # Form Feedback
        for i, message in enumerate(self.feedback_messages[:3]):
            color = self.colors['warning'] if "!" in message else self.colors['good']
            frame = self.put_ru_text(frame, f"- {message}", (20, y_offset),
                                     font_size='small', color=color)
            y_offset += line_height - 5

        # Knee Analysis
        y_offset += 5
        for i, message in enumerate(self.knee_feedback[:2]):
            color = self.colors['warning'] if "!" in message else self.colors['good']
            frame = self.put_ru_text(frame, f"- {message}", (20, y_offset),
                                     font_size='small', color=color)
            y_offset += line_height - 5

        return frame

    def process_frame(self, frame):
        """Process single frame"""
        # Perform prediction
        results = self.model(
            frame,
            conf=self.conf_threshold,
            imgsz=320,
            iou=0.5,
            max_det=1,
            verbose=False
        )

        # Visualize results
        annotated_frame = self.draw_body_points(results[0], frame)

        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time

        # Add information to frame
        annotated_frame = self.add_fitness_info(annotated_frame, results[0])

        return annotated_frame

    def reset_counter(self):
        """Reset repetition counter"""
        self.rep_count = 0
        self.min_knee_angle = 180

# Инициализация трекера
trainer = SquatTrainer()

def process_frames():
    """Функция для обработки кадров в отдельном потоке"""
    global current_frame, processing_active
    
    while processing_active:
        try:
            # Получаем кадр из очереди (с таймаутом чтобы не блокировать навсегда)
            frame = frame_queue.get(timeout=1.0)
            current_frame = frame
            
            # Обрабатываем кадр
            processed_frame = trainer.process_frame(frame)
            
            # Кодируем обработанный кадр в JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Отправляем результат
            if result_queue.empty():
                result_queue.put(processed_frame_data)
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

# Запускаем поток обработки
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Принимает кадры с телефона"""
    global current_frame
    
    try:
        # Получаем изображение из запроса
        image_data = request.json['image']
        
        # Декодируем base64
        image_data = image_data.split(',')[1]  # Убираем префикс data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Добавляем кадр в очередь для обработки
            if frame_queue.empty():
                frame_queue.put(frame)
            
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})
            
    except Exception as e:
        print(f"Error uploading frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_processed_frame')
def get_processed_frame():
    """Возвращает обработанный кадр"""
    try:
        # Получаем обработанный кадр из очереди
        processed_frame_data = result_queue.get(timeout=2.0)
        return jsonify({
            'status': 'success',
            'image': f"data:image/jpeg;base64,{processed_frame_data}",
            'rep_count': trainer.rep_count,
            'squat_depth': round(trainer.squat_depth, 1),
            'feedback': trainer.feedback_messages[:2] + trainer.knee_feedback[:1]
        })
    except queue.Empty:
        return jsonify({'status': 'no_frame'})
    except Exception as e:
        print(f"Error getting processed frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset', methods=['POST'])
def reset_counter():
    """Сброс счетчика повторений"""
    trainer.reset_counter()
    return jsonify({'status': 'success', 'rep_count': trainer.rep_count})

@app.route('/stats')
def get_stats():
    """Возвращает текущую статистику"""
    return jsonify({
        'rep_count': trainer.rep_count,
        'squat_depth': round(trainer.squat_depth, 1),
        'feedback': trainer.feedback_messages[:2] + trainer.knee_feedback[:1],
        'view_angle': trainer.view_angle
    })

if __name__ == '__main__':
    print("Запуск сервера тренера приседаний...")
    print("Откройте в браузере телефона: http://ВАШ_IP:5000")
    print("Убедитесь, что телефон и компьютер в одной сети Wi-Fi!")
    
    # Запускаем сервер на всех интерфейсах
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
