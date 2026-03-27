import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Константы

# Минимальная доля пересечения bbox человека с зоной столика,
# чтобы считать человека «у столика» (intersection / area_bbox).
OVERLAP_THRESHOLD = 0.25

# Сколько подряд одинаковых кадров нужно для подтверждения смены состояния.
STATE_CONFIRM_FRAMES = 5

# Цвета рамки столика (BGR).
COLOR_EMPTY = (0, 200, 0)       # зелёный
COLOR_OCCUPIED = (0, 0, 220)    # красный
COLOR_APPROACH = (0, 180, 255)  # оранжевый

# YOLO: класс «person» = 0 в COCO, порог уверенности.
PERSON_CLASS_ID = 0
YOLO_CONF = 0.3


# Детектор людей

class YOLOPersonDetector:
    """Детекция людей через YOLOv11n (предобученная модель COCO)."""

    def __init__(self):
        print("[INFO] Загрузка YOLO11n...")
        # Использую medium модель, потому что small и nano не совсем хорошо
        # справляются с детекцией.
        self.model = YOLO("yolo8m.pt")
        print("[INFO] Модель загружена")

    def detect(self, frame: np.ndarray) -> list:
        """Возвращает список bbox обнаруженных людей."""
        results = self.model(frame, verbose=False, conf=YOLO_CONF)
        boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes


# Вспомогательные функции

def compute_overlap(box_a, box_b) -> float:
    """
    Доля площади пересечения box_a ∩ box_b относительно площади box_a.
    Оба box в формате (x, y, w, h).
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a = aw * ah
    if area_a == 0:
        return 0.0
    return inter_area / area_a


def get_roi(video_path: str, roi_arg) -> tuple:
    """
    Получить координаты зоны столика.
    Если roi_arg задан — парсим строку «x,y,w,h».
    Иначе — открываем интерактивный выбор через cv2.selectROI.
    """
    if roi_arg:
        parts = [int(x.strip()) for x in roi_arg.split(",")]
        if len(parts) != 4:
            sys.exit("[ERROR] --roi должен быть в формате x,y,w,h")
        print(
            f"[INFO] ROI из аргументов: x={parts[0]}, y={parts[1]}, "
            f"w={parts[2]}, h={parts[3]}"
        )
        return tuple(parts)

    # Интерактивный выбор на первом кадре.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Не удалось открыть видео: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("[ERROR] Не удалось прочитать первый кадр")

    print("\n[INFO] Выберите зону столика мышкой и нажмите ENTER / SPACE.")
    print("       Для отмены — нажмите C.\n")
    roi = cv2.selectROI(
        "Выберите столик", frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        sys.exit("[ERROR] Зона столика не выбрана")

    print(f"[INFO] ROI столика: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    return roi


# Машина состояний столика

class TableStateMachine:
    """
    Три состояния:
      EMPTY    — в зоне столика нет людей
      OCCUPIED — есть хотя бы один человек
      APPROACH — момент перехода EMPTY → OCCUPIED (подход к столу)
    """

    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    APPROACH = "APPROACH"

    def __init__(self, fps: float):
        self.state = self.EMPTY
        self.fps = fps

        # Механизм подтверждения.
        self._pending_state = None
        self._pending_count = 0

        # Журнал событий.
        self.events = []

    def update(self, frame_no: int, person_in_zone: bool) -> str:
        """Обновляет состояние. Возвращает текущее состояние."""
        raw = self.OCCUPIED if person_in_zone else self.EMPTY

        if raw != self.state and raw != self.APPROACH:
            if self._pending_state == raw:
                self._pending_count += 1
            else:
                self._pending_state = raw
                self._pending_count = 1

            if self._pending_count >= STATE_CONFIRM_FRAMES:
                self._transition(frame_no, raw)
                self._pending_state = None
                self._pending_count = 0
        else:
            self._pending_state = None
            self._pending_count = 0

        return self.state

    def _transition(self, frame_no: int, new_state: str):
        """Фиксирует переход между состояниями и пишет в журнал."""
        ts = frame_no / self.fps if self.fps > 0 else 0.0

        if self.state == self.EMPTY and new_state == self.OCCUPIED:
            # Фиксируем событие APPROACH, затем переходим в OCCUPIED.
            self.events.append({
                "frame": frame_no,
                "time_sec": round(ts, 2),
                "event": self.APPROACH,
            })
            self.state = self.OCCUPIED
            self.events.append({
                "frame": frame_no,
                "time_sec": round(ts, 2),
                "event": self.OCCUPIED,
            })

        elif self.state == self.OCCUPIED and new_state == self.EMPTY:
            self.state = self.EMPTY
            self.events.append({
                "frame": frame_no,
                "time_sec": round(ts, 2),
                "event": self.EMPTY,
            })

    def get_color(self) -> tuple:
        """Цвет рамки в зависимости от состояния."""
        if self.state == self.EMPTY:
            return COLOR_EMPTY
        elif self.state == self.OCCUPIED:
            return COLOR_OCCUPIED
        return COLOR_APPROACH


# Аналитика

def build_analytics(events: list) -> pd.DataFrame:
    """
    Строит DataFrame с событиями и рассчитывает время ожидания:
    от EMPTY (гость ушёл) до следующего APPROACH (новый подошёл).
    """
    df = pd.DataFrame(events)
    if df.empty:
        return df

    df["wait_time_sec"] = np.nan
    for i in range(len(df)):
        if df.iloc[i]["event"] == "EMPTY":
            for j in range(i + 1, len(df)):
                if df.iloc[j]["event"] == "APPROACH":
                    wait = df.iloc[j]["time_sec"] - df.iloc[i]["time_sec"]
                    df.at[i, "wait_time_sec"] = round(wait, 2)
                    break
    return df


def print_stats(df: pd.DataFrame):
    """Выводит базовую статистику в консоль."""
    print("\n" + "=" * 50)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 50)

    if df.empty:
        print("  Событий не зафиксировано.")
        return

    print(f"  Всего событий: {len(df)}")
    print(f"    EMPTY:    {(df['event'] == 'EMPTY').sum()}")
    print(f"    APPROACH: {(df['event'] == 'APPROACH').sum()}")
    print(f"    OCCUPIED: {(df['event'] == 'OCCUPIED').sum()}")

    waits = df["wait_time_sec"].dropna()
    if not waits.empty:
        print(f"\n  Среднее время между уходом гостя")
        print(f"  и подходом следующего: {waits.mean():.1f} сек")
        print(f"  Мин: {waits.min():.1f} сек | Макс: {waits.max():.1f} сек")
    else:
        print("\n  Недостаточно данных для расчёта задержки.")

    print("=" * 50)


# Основной пайплайн обработки видео

def process_video(video_path: str, roi: tuple, output_path: str):
    """
    Обрабатывает видео кадр за кадром:
    1. Детекция людей через YOLO
    2. Проверка пересечения bbox людей с зоной столика
    3. Обновление машины состояний
    4. Отрисовка рамки столика с цветом по состоянию
    5. Запись результата в output.mp4
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Видео: {w}x{h}, {fps:.1f} FPS, {total} кадров")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    detector = YOLOPersonDetector()
    sm = TableStateMachine(fps)

    rx, ry, rw, rh = roi
    frame_no = 0

    print("[INFO] Обработка видео...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # Детекция людей.
        person_boxes = detector.detect(frame)

        # Проверяем, есть ли человек в зоне столика.
        person_in_zone = any(
            compute_overlap(box, roi) >= OVERLAP_THRESHOLD
            for box in person_boxes
        )

        # Обновляем состояние столика.
        state = sm.update(frame_no, person_in_zone)

        # Рисуем bounding box столика с цветом состояния.
        color = sm.get_color()
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)
        cv2.putText(
            frame, state, (rx, ry - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        writer.write(frame)

        if frame_no % 500 == 0:
            print(f"  {frame_no / total * 100:.0f}% ({frame_no}/{total})")

    cap.release()
    writer.release()
    print(f"[INFO] Готово. Результат: {output_path}")

    return sm.events


# Точка входа

def main():
    parser = argparse.ArgumentParser(
        description="Прототип детекции уборки столиков по видео"
    )
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument("--output", default="output/output.mp4", help="Путь к выходному видео")
    parser.add_argument("--roi", default=None, help="Координаты столика: x,y,w,h")
    args = parser.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"[ERROR] Файл не найден: {args.video}")

    roi = get_roi(args.video, args.roi)
    events = process_video(args.video, roi, args.output)
    df = build_analytics(events)
    print_stats(df)


if __name__ == "__main__":
    main()