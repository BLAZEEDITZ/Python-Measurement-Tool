import cv2
import numpy as np
import math
from datetime import datetime
import json
import platform

try:
    import tkinter as tk
except ImportError:
    tk = None


class MeasurementTool:
    def __init__(self):
        self.detected_objects = []
        self.current_frame = None
        self.measurements = []
        self.alert_message = ""
        self.alert_time = 0
        self.alert_color = (0, 255, 0)
        self.selected_index = None
        self.auto_mode_enabled = False

        print("\nDetecting system specifications...")
        self.detect_monitor_specs()
        self.detect_camera_specs()
        self.calculate_calibration()

        self.min_area = 500
        self.max_area = 500000
        self.contour_threshold = 50

    def detect_monitor_specs(self):
        try:
            if tk is not None:
                root = tk.Tk()
                root.withdraw()

                self.screen_width = root.winfo_screenwidth()
                self.screen_height = root.winfo_screenheight()
                self.dpi = root.winfo_fpixels('1i')

                diagonal_pixels = math.sqrt(self.screen_width ** 2 + self.screen_height ** 2)
                self.monitor_diagonal_inches = diagonal_pixels / self.dpi

                root.destroy()

                print("Monitor Detected: {}x{} at {:.0f} DPI".format(
                    self.screen_width, self.screen_height, self.dpi))
            else:
                self.use_default_monitor_specs()

        except Exception as e:
            print("Monitor detection failed: {}".format(str(e)))
            self.use_default_monitor_specs()

    def use_default_monitor_specs(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.dpi = 96.0
        diagonal_pixels = math.sqrt(self.screen_width ** 2 + self.screen_height ** 2)
        self.monitor_diagonal_inches = diagonal_pixels / self.dpi
        print("Using default: 1920x1080 at 96 DPI")

    def detect_camera_specs(self):
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise Exception("Camera not available")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_fps = int(cap.get(cv2.CAP_PROP_FPS))

            print("Camera Detected: {}x{} at {} FPS".format(
                self.camera_width, self.camera_height, self.camera_fps))

            cap.release()

        except Exception as e:
            print("Camera detection failed: {}".format(str(e)))
            self.camera_width = 1280
            self.camera_height = 720
            self.camera_fps = 30

    def calculate_calibration(self):
        self.pixels_per_cm = self.dpi / 2.54
        print("Calibration: {:.4f} pixels per cm\n".format(self.pixels_per_cm))

    def show_alert(self, message, color=(0, 255, 100)):
        self.alert_message = message
        self.alert_time = datetime.now()
        self.alert_color = color

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.check_button_click(x, y)

    def check_button_click(self, x, y):
        button_y = 20
        button_height = 50
        button_spacing = 10
        button_width = 120
        button_x_start = 20

        buttons = [
            ('DETECT', self.detect_objects),
            ('ANALYZE', self.analyze_selected),
            ('AUTO', self.toggle_auto),
            ('CLEAR', self.clear_detections),
            ('SAVE', self.save_measurements),
            ('INFO', self.show_info),
            ('EXIT', lambda: None)
        ]

        for idx, (label, callback) in enumerate(buttons):
            bx = button_x_start + idx * (button_width + button_spacing)
            by = button_y

            if bx <= x <= bx + button_width and by <= y <= by + button_height:
                if label != 'EXIT':
                    callback()
                return True

        return False

    def detect_objects(self):
        if self.current_frame is None:
            self.show_alert("No camera feed", (0, 0, 255))
            return

        frame = self.current_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blur, self.contour_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.detected_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                obj = self.extract_object_info(contour, area)
                self.detected_objects.append(obj)

        if self.detected_objects:
            self.detected_objects.sort(key=lambda x: x['area_cm2'], reverse=True)
            self.selected_index = 0
            self.show_alert("Detected {} objects".format(len(self.detected_objects)), (0, 255, 100))
        else:
            self.show_alert("No objects found", (255, 165, 0))

    def approximate_polygon(self, contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.02 * peri, True)

    def extract_object_info(self, contour, area_px):
        perimeter_px = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        area_cm2 = area_px / (self.pixels_per_cm ** 2)
        perimeter_cm = perimeter_px / self.pixels_per_cm
        width_cm = w / self.pixels_per_cm
        height_cm = h / self.pixels_per_cm

        circularity = 4 * math.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0

        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

        approx = self.approximate_polygon(contour)
        vertices = len(approx)

        return {
            'contour': contour,
            'area_px': area_px,
            'area_cm2': area_cm2,
            'perimeter_px': perimeter_px,
            'perimeter_cm': perimeter_cm,
            'bbox': (x, y, w, h),
            'rotated_box': box,
            'width_px': w,
            'height_px': h,
            'width_cm': width_cm,
            'height_cm': height_cm,
            'aspect_ratio': max(width_cm, height_cm) / min(width_cm, height_cm) if min(width_cm, height_cm) > 0 else 0,
            'circularity': circularity,
            'centroid': (cx, cy),
            'vertices': vertices,
            'timestamp': datetime.now().isoformat()
        }

    def classify_shape(self, obj):
        circularity = obj['circularity']
        aspect_ratio = obj['aspect_ratio']
        vertices = obj['vertices']

        if circularity > 0.88:
            return "Circle"
        elif vertices == 3:
            return "Triangle"
        elif vertices == 4:
            if 0.85 < aspect_ratio < 1.15:
                return "Square"
            else:
                return "Rectangle"
        elif vertices == 5:
            return "Pentagon"
        elif vertices == 6:
            return "Hexagon"
        elif vertices > 6:
            if circularity > 0.70:
                return "Polygon"
            else:
                return "Irregular"
        else:
            return "Shape"

    def analyze_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.detected_objects):
            self.show_alert("No object selected", (0, 0, 255))
            return

        obj = self.detected_objects[self.selected_index]

        measurement = {
            'object_id': len(self.measurements) + 1,
            'width_cm': round(obj['width_cm'], 2),
            'height_cm': round(obj['height_cm'], 2),
            'area_cm2': round(obj['area_cm2'], 2),
            'perimeter_cm': round(obj['perimeter_cm'], 2),
            'shape': self.classify_shape(obj),
            'vertices': obj['vertices'],
            'timestamp': datetime.now().isoformat()
        }

        self.measurements.append(measurement)
        msg = "Measured: {:.2f}cm x {:.2f}cm".format(obj['width_cm'], obj['height_cm'])
        self.show_alert(msg, (0, 255, 100))

    def toggle_auto(self):
        self.auto_mode_enabled = not self.auto_mode_enabled
        status = "ON" if self.auto_mode_enabled else "OFF"
        self.show_alert("Auto Mode: {}".format(status), (0, 200, 255))

    def clear_detections(self):
        self.detected_objects = []
        self.selected_index = None
        self.show_alert("Cleared", (200, 200, 200))

    def save_measurements(self):
        if not self.measurements:
            self.show_alert("No measurements", (0, 0, 255))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "measurements_{}.json".format(timestamp)

        data = {
            'system': {
                'monitor': "{}x{}@{:.0f}DPI".format(self.screen_width, self.screen_height, self.dpi),
                'camera': "{}x{}@{}FPS".format(self.camera_width, self.camera_height, self.camera_fps),
                'pixels_per_cm': round(self.pixels_per_cm, 4)
            },
            'measurements': self.measurements,
            'total': len(self.measurements)
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.show_alert("Saved: {}".format(filename), (0, 255, 100))
        except Exception as e:
            self.show_alert("Error: {}".format(str(e)), (0, 0, 255))

    def show_info(self):
        info = "\n" + "="*60
        info += "\nMEASUREMENT TOOL INFO\n"
        info += "="*60
        info += "\nMONITOR: {}x{} @ {:.0f} DPI".format(
            self.screen_width, self.screen_height, self.dpi)
        info += "\nCAMERA: {}x{} @ {} FPS".format(
            self.camera_width, self.camera_height, self.camera_fps)
        info += "\nCALIBRATION: {:.4f} pixels/cm".format(self.pixels_per_cm)
        info += "\nMEASUREMENTS: {}".format(len(self.measurements))
        info += "\n" + "="*60 + "\n"
        print(info)
        self.show_alert("Info displayed", (100, 200, 255))

    def draw_frame(self, frame):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 0), (w, 85), (100, 100, 100), 2)

        button_y = 20
        button_height = 50
        button_spacing = 10
        button_width = 120
        button_x_start = 20

        buttons = [
            ('DETECT', (0, 200, 100)),
            ('ANALYZE', (100, 100, 200)),
            ('AUTO', (200, 100, 0) if self.auto_mode_enabled else (100, 100, 100)),
            ('CLEAR', (100, 100, 200)),
            ('SAVE', (100, 100, 200)),
            ('INFO', (100, 100, 200)),
            ('EXIT', (200, 50, 50))
        ]

        for idx, (label, color) in enumerate(buttons):
            bx = button_x_start + idx * (button_width + button_spacing)
            by = button_y

            cv2.rectangle(frame, (bx, by), (bx + button_width, by + button_height), color, -1)
            cv2.rectangle(frame, (bx, by), (bx + button_width, by + button_height), (255, 255, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.45, 1)[0]
            text_x = bx + (button_width - text_size[0]) // 2
            text_y = by + (button_height + text_size[1]) // 2

            cv2.putText(frame, label, (text_x, text_y), font, 0.45, (255, 255, 255), 1)

        for i, obj in enumerate(self.detected_objects):
            x, y, w_obj, h_obj = obj['bbox']

            is_selected = (i == self.selected_index)
            color = (0, 255, 0) if is_selected else (0, 165, 255)
            thickness = 3 if is_selected else 2

            cv2.rectangle(frame, (x, y), (x + w_obj, y + h_obj), color, thickness)
            cv2.drawContours(frame, [obj['contour']], 0, color, 2)

            cx, cy = obj['centroid']
            cv2.circle(frame, (cx, cy), 5, color, -1)

            label = "#{}:{:.1f}x{:.1f}cm".format(i + 1, obj['width_cm'], obj['height_cm'])
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.selected_index is not None and self.selected_index < len(self.detected_objects):
            obj = self.detected_objects[self.selected_index]

            panel_x = 20
            panel_y = h - 220
            panel_w = 320
            panel_h = 200

            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 100), 2)

            cv2.putText(frame, "Object Measurements (CM)", (panel_x + 10, panel_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            y_offset = panel_y + 50
            info_lines = [
                "Length: {:.2f} cm".format(obj['width_cm']),
                "Height: {:.2f} cm".format(obj['height_cm']),
                "Area: {:.2f} cm2".format(obj['area_cm2']),
                "Perimeter: {:.2f} cm".format(obj['perimeter_cm']),
                "Shape: {}".format(self.classify_shape(obj)),
                "Vertices: {}".format(obj['vertices'])
            ]

            for line in info_lines:
                cv2.putText(frame, line, (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25

        if self.alert_message:
            if (datetime.now() - self.alert_time).total_seconds() < 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(self.alert_message, font, 0.8, 2)[0]

                alert_x = (w - text_size[0]) // 2
                alert_y = h - 30

                cv2.rectangle(frame, (alert_x - 10, alert_y - 25), (alert_x + text_size[0] + 10, alert_y + 5),
                             (20, 20, 20), -1)
                cv2.rectangle(frame, (alert_x - 10, alert_y - 25), (alert_x + text_size[0] + 10, alert_y + 5),
                             self.alert_color, 2)

                cv2.putText(frame, self.alert_message, (alert_x, alert_y),
                           font, 0.8, self.alert_color, 2)

        mode_text = "AUTO" if self.auto_mode_enabled else "MANUAL"
        cv2.putText(frame, "Mode: {} | Objects: {} | Measured: {}".format(
            mode_text, len(self.detected_objects), len(self.measurements)),
                   (w - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        window_name = 'Measurement Tool'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height + 100)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\nTool ready. Click buttons to start.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.current_frame = frame.copy()

            if self.auto_mode_enabled:
                self.detect_objects()

            display_frame = self.draw_frame(frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.measurements:
            self.print_summary()

    def print_summary(self):
        summary = "\n" + "="*60
        summary += "\nMEASUREMENT SUMMARY"
        summary += "\n" + "="*60
        summary += "\nTotal Measurements: {}\n".format(len(self.measurements))

        for m in self.measurements:
            summary += "\nObject #{}:".format(m['object_id'])
            summary += "\n  Length: {:.2f} cm".format(m['width_cm'])
            summary += "\n  Height: {:.2f} cm".format(m['height_cm'])
            summary += "\n  Area: {:.2f} cm2".format(m['area_cm2'])
            summary += "\n  Perimeter: {:.2f} cm".format(m['perimeter_cm'])
            summary += "\n  Shape: {}".format(m['shape'])
            summary += "\n  Vertices: {}".format(m['vertices'])

        summary += "\n" + "="*60 + "\n"
        print(summary)


def main():
    print("\n" + "="*60)
    print("MEASUREMENT TOOL v5.0 - GEOMETRIC SHAPE DETECTION")
    print("="*60)
    tool = MeasurementTool()
    tool.run()


if __name__ == "__main__":
    main()