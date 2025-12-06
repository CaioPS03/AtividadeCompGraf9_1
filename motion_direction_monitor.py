import cv2
import numpy as np
import argparse
import time
from collections import deque


class BackgroundModel:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.background = None

    def apply(self, frame_gray):
        """Atualiza o background usando média acumulada e retorna o background atual."""
        frame_f = frame_gray.astype('float32')
        if self.background is None:
            self.background = frame_f.copy()
        else:
            cv2.accumulateWeighted(frame_f, self.background, self.alpha)
        return cv2.convertScaleAbs(self.background)


class MotionDetector:
    def __init__(self, min_area=1500):
        self.min_area = min_area

    def detect(self, frame_gray, background_gray):
        # diferença absoluta (isso reduz efeito de mudança lenta de iluminação quando alpha pequeno)
        diff = cv2.absdiff(background_gray, frame_gray)
        # Suavizar para reduzir ruído de alta frequência
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        # Threshold adaptativo simples (otsu)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remover pequenas regiões com morfologia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Encontrar contornos
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        return clean, valid_contours


class ObjectTracker:
    def __init__(self, history=8, area_threshold=0.08):
        self.history = history
        self.area_history = deque(maxlen=history)
        self.area_threshold = area_threshold

    def update(self, contour):
        if contour is None:
            self.area_history.append(0)
            return None
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        self.area_history.append(area)
        direction = self.infer_direction()
        speed = self.estimate_speed()
        return {'area': area, 'bbox': (x, y, w, h), 'direction': direction, 'speed': speed}

    def infer_direction(self):
        # usando média dos últimos half vs first half
        if len(self.area_history) < 2:
            return 'unknown'
        arr = np.array(self.area_history, dtype=np.float32)
        # if all zeros (no object) -> unknown
        if np.all(arr == 0):
            return 'no_object'
        half = len(arr) // 2 or 1
        prev_mean = np.mean(arr[:half])
        next_mean = np.mean(arr[half:])
        if prev_mean == 0:
            # object just appeared
            if next_mean > 0:
                return 'approaching' if next_mean > prev_mean else 'unknown'
            return 'unknown'
        relative_change = (next_mean - prev_mean) / (prev_mean + 1e-6)
        if relative_change > self.area_threshold:
            return 'approaching'
        elif relative_change < -self.area_threshold:
            return 'receding'
        else:
            return 'stationary'

    def estimate_speed(self):
        # simple: diff between last two non-zero areas normalized by frame
        if len(self.area_history) < 2:
            return 0.0
        arr = np.array(self.area_history, dtype=np.float32)
        # use last two
        last = arr[-1]
        prev = arr[-2]
        # avoid division by zero
        return float(last - prev)


class EdgeProcessor:
    def __init__(self):
        pass

    def process(self, roi_color):
        if roi_color is None or roi_color.size == 0:
            return None, None
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        # Sobel
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        # Canny - auto thresholds from median
        v = np.median(roi_gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        if lower >= upper:
            lower = int(max(0, v * 0.5))
            upper = int(min(255, v * 1.5))
        canny = cv2.Canny(roi_gray, lower, upper)
        return sobel, canny


class Visualizer:
    def __init__(self, show_windows=True):
        self.show_windows = show_windows

    def draw(self, frame, mask, obj_info, sobel, canny):
        disp = frame.copy()
        h, w = frame.shape[:2]
        # draw mask small at corner
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (w//4, h//4))
        disp[0: h//4, 0: w//4] = mask_small

        # draw bbox and info
        if obj_info is not None:
            x, y, bw, bh = obj_info['bbox']
            area = obj_info['area']
            direction = obj_info['direction']
            speed = obj_info['speed']
            # bbox
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # centroid
            cx = int(x + bw / 2)
            cy = int(y + bh / 2)
            cv2.circle(disp, (cx, cy), 4, (0, 255, 0), -1)
            # text
            text = f"Area: {int(area)} | Dir: {direction} | dArea: {int(speed)}"
            cv2.putText(disp, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # show edges and canny to the right as thumbnails
        if sobel is not None:
            sobel_col = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR) if len(sobel.shape) == 2 else sobel
            canny_col = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR) if len(canny.shape) == 2 else canny
            th_h = h // 4
            th_w = int(th_h * (sobel.shape[1] / sobel.shape[0])) if sobel.shape[0] else th_h
            sob_res = cv2.resize(sobel_col, (th_w, th_h))
            can_res = cv2.resize(canny_col, (th_w, th_h))
            # put them stacked on top-right
            x0 = w - th_w - 10
            disp[10:10+th_h, x0:x0+th_w] = sob_res
            disp[20+th_h:20+2*th_h, x0:x0+th_w] = can_res

        if self.show_windows:
            cv2.imshow('Monitor - Live', disp)


def main():
    parser = argparse.ArgumentParser(description='Motion direction monitor')
    parser.add_argument('--source', type=str, default='0', help='camera index or video file path')
    parser.add_argument('--min-area', type=int, default=1500)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--history', type=int, default=8)
    parser.add_argument('--area-threshold', type=float, default=0.08)
    args = parser.parse_args()

    # configure capture
    try:
        src = int(args.source)
    except Exception:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Erro ao abrir a fonte de vídeo:', args.source)
        return

    bg = BackgroundModel(alpha=args.alpha)
    detector = MotionDetector(min_area=args.min_area)
    tracker = ObjectTracker(history=args.history, area_threshold=args.area_threshold)
    edges = EdgeProcessor()
    viz = Visualizer(show_windows=True)

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # resize to make processing faster (optional)
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update background and get background image
        background = bg.apply(gray)

        # detect motion
        mask, contours = detector.detect(gray, background)

        # choose the largest contour as object (if any)
        contour = None
        if contours:
            contour = max(contours, key=cv2.contourArea)

        obj_info = tracker.update(contour)

        sobel_img, canny_img = None, None
        if obj_info is not None and obj_info['area'] > 0:
            x, y, bw, bh = obj_info['bbox']
            # expand bbox a bit for context
            pad = 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + bw + pad)
            y1 = min(frame.shape[0], y + bh + pad)
            roi = frame[y0:y1, x0:x1]
            sobel_img, canny_img = edges.process(roi)
        
        viz.draw(frame, mask, obj_info, sobel_img, canny_img)

        # handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            # pause
            while True:
                k = cv2.waitKey(0) or 0
                if k == ord('p'):
                    break
                if k == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
