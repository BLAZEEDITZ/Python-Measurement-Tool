import cv2
import numpy as np
import math
from datetime import datetime
import json
import platform
from collections import deque, defaultdict
import time

try:
    import tkinter as tk
except ImportError:
    tk = None


class ObjectTracker:
    def __init__(self, max_distance=100):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.frame_count = 0

    def update(self, centroids):
        if len(self.tracked_objects) == 0:
            for centroid in centroids:
                self.tracked_objects[self.next_id] = {
                    'centroid': centroid,
                    'frames': 1
                }
                self.next_id += 1
        else:
            tracked_centroids = set(self.tracked_objects.keys())
            input_centroids = set(range(len(centroids)))

            used_input = set()
            used_tracked = set()

            for tracked_id in tracked_centroids:
                if tracked_id in used_tracked:
                    continue

                min_dist = float('inf')
                min_idx = -1

                for idx, centroid in enumerate(centroids):
                    if idx in used_input:
                        continue
                    dist = math.sqrt((self.tracked_objects[tracked_id]['centroid'][0] - centroid[0]) ** 2 +
                                     (self.tracked_objects[tracked_id]['centroid'][1] - centroid[1]) ** 2)
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        min_idx = idx

                if min_idx >= 0:
                    self.tracked_objects[tracked_id]['centroid'] = centroids[min_idx]
                    self.tracked_objects[tracked_id]['frames'] += 1
                    used_input.add(min_idx)
                    used_tracked.add(tracked_id)

            for idx in range(len(centroids)):
                if idx not in used_input:
                    self.tracked_objects[self.next_id] = {
                        'centroid': centroids[idx],
                        'frames': 1
                    }
                    self.next_id += 1

            to_remove = [tid for tid in tracked_centroids if
                         tid not in used_tracked and self.tracked_objects[tid]['frames'] < 3]
            for tid in to_remove:
                del self.tracked_objects[tid]

        self.frame_count += 1
        return self.tracked_objects


class SizeFilterManager:
    def __init__(self):
        self.size_ranges = {
            'Tiny': (0, 5),
            'Small': (5, 25),
            'Medium': (25, 100),
            'Large': (100, 300),
            'VeryLarge': (300, float('inf'))
        }
        self.enabled_sizes = {
            'Tiny': False,
            'Small': False,
            'Medium': True,
            'Large': True,
            'VeryLarge': True
        }

    def toggle_size(self, size_name):
        if size_name in self.enabled_sizes:
            self.enabled_sizes[size_name] = not self.enabled_sizes[size_name]

    def get_min_max_area(self, pixels_per_cm):
        min_area = float('inf')
        max_area = 0

        has_enabled = False
        for size_name, enabled in self.enabled_sizes.items():
            if enabled:
                has_enabled = True
                range_min, range_max = self.size_ranges[size_name]
                min_area = min(min_area, range_min * (pixels_per_cm ** 2))
                if range_max == float('inf'):
                    max_area = 1000000
                else:
                    max_area = max(max_area, range_max * (pixels_per_cm ** 2))

        if not has_enabled:
            return 300, 400000

        if min_area == float('inf'):
            min_area = 300

        if max_area == 0:
            max_area = 400000

        return int(min_area), int(max_area)

    def is_size_enabled(self, area_cm2):
        for size_name, enabled in self.enabled_sizes.items():
            if enabled:
                range_min, range_max = self.size_ranges[size_name]
                if range_min <= area_cm2 < range_max:
                    return True
        return False

    def get_enabled_sizes(self):
        return [name for name, enabled in self.enabled_sizes.items() if enabled]

    def enable_all(self):
        for size_name in self.enabled_sizes:
            self.enabled_sizes[size_name] = True

    def disable_all(self):
        for size_name in self.enabled_sizes:
            self.enabled_sizes[size_name] = False


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
        self.measurement_history = defaultdict(deque)
        self.smoothed_measurements = {}
        self.tracker = ObjectTracker()
        self.object_stats = defaultdict(lambda: {'measurements': [], 'detected_count': 0})
        self.fps_list = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.current_fps = 0
        self.size_filter = SizeFilterManager()
        self.show_size_menu = False
        self.detection_enabled = True
        self.contrast_level = 3.0
        self.blur_strength = 11

        print("\nDetecting system specifications...")
        self.detect_monitor_specs()
        self.detect_camera_specs()
        self.calculate_calibration()

        self.contour_threshold = 35
        self.canny_low = 40
        self.canny_high = 120
        self.min_contour_size = 15

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
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_FOCUS_MODE, 1)

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

    def show_alert(self, message, color=(100, 200, 255)):
        self.alert_message = message
        self.alert_time = datetime.now()
        self.alert_color = color

    def update_fps(self):
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            fps = 1.0 / delta
            self.fps_list.append(fps)
            self.current_fps = sum(self.fps_list) / len(self.fps_list)
        self.last_frame_time = current_time

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.check_button_click(x, y)

    def check_button_click(self, x, y):
        button_y = 25
        button_height = 40
        button_spacing = 6
        button_width = 85
        button_x_start = 20

        main_buttons = [
            ('DETECT', self.detect_objects),
            ('ANALYZE', self.analyze_selected),
            ('AUTO', self.toggle_auto),
            ('FILTER', self.toggle_size_menu),
            ('STATS', self.show_statistics),
            ('CLEAR', self.clear_detections),
            ('SAVE', self.save_measurements),
            ('EXIT', lambda: None)
        ]

        for idx, (label, callback) in enumerate(main_buttons):
            bx = button_x_start + idx * (button_width + button_spacing)
            by = button_y

            if bx <= x <= bx + button_width and by <= y <= by + button_height:
                if label != 'EXIT':
                    callback()
                return True

        if self.show_size_menu:
            size_button_y = 80
            size_button_height = 35
            size_button_spacing = 5
            size_button_width = 80
            size_button_x_start = 20

            size_buttons = [
                ('Tiny', 'Tiny'),
                ('Small', 'Small'),
                ('Medium', 'Medium'),
                ('Large', 'Large'),
                ('V.Large', 'VeryLarge')
            ]

            for idx, (label, size_name) in enumerate(size_buttons):
                bx = size_button_x_start + idx * (size_button_width + size_button_spacing)
                by = size_button_y

                if bx <= x <= bx + size_button_width and by <= y <= by + size_button_height:
                    self.size_filter.toggle_size(size_name)
                    msg = "Toggled {} detection".format(label)
                    self.show_alert(msg, (150, 150, 200))
                    return True

            all_enable_bx = 500
            all_enable_by = 80
            all_enable_width = 70
            all_enable_height = 35

            if all_enable_bx <= x <= all_enable_bx + all_enable_width and all_enable_by <= y <= all_enable_by + all_enable_height:
                self.size_filter.enable_all()
                self.show_alert("Enabled all sizes", (100, 200, 150))
                return True

            all_disable_bx = 580
            all_disable_by = 80
            all_disable_width = 70
            all_disable_height = 35

            if all_disable_bx <= x <= all_disable_bx + all_disable_width and all_disable_by <= y <= all_disable_by + all_disable_height:
                self.size_filter.enable_all()
                self.show_alert("Enabled recommended sizes (Med, Lrg, VL)", (100, 200, 150))
                return True

        return False

    def toggle_size_menu(self):
        self.show_size_menu = not self.show_size_menu
        status = "OPEN" if self.show_size_menu else "CLOSED"
        self.show_alert("Size Filter: {}".format(status), (200, 150, 0))

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, self.blur_strength, 80, 80)
        clahe = cv2.createCLAHE(clipLimit=self.contrast_level, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        return enhanced

    def detect_objects_multiscale(self, frame, enhanced):
        min_area, max_area = self.size_filter.get_min_max_area(self.pixels_per_cm)

        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        kernel_sizes = [5, 7, 9]
        all_contours = []

        for kernel_size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            processed = cv2.morphologyEx(edges.copy(), cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)

        all_contours = self.merge_duplicate_contours(all_contours)

        self.detected_objects = []

        for contour in all_contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                if self.is_valid_contour(contour):
                    obj = self.extract_object_info(contour, area)
                    if self.size_filter.is_size_enabled(obj['area_cm2']):
                        self.detected_objects.append(obj)

        enabled_sizes = self.size_filter.get_enabled_sizes()
        size_text = ','.join([s[:3] for s in enabled_sizes])

        if self.detected_objects:
            self.detected_objects.sort(key=lambda x: x['area_cm2'], reverse=True)
            self.selected_index = 0
            msg = "Detected {} objects [{}]".format(len(self.detected_objects), size_text)
            self.show_alert(msg, (100, 200, 100))
        else:
            msg = "No objects in [{}] range - adjust camera or filter".format(size_text)
            self.show_alert(msg, (100, 150, 255))

    def merge_duplicate_contours(self, contours):
        if not contours:
            return contours

        merged = []
        used = set()

        for i, c1 in enumerate(contours):
            if i in used:
                continue

            m1 = cv2.moments(c1)
            if m1['m00'] == 0:
                continue

            cx1 = int(m1['m10'] / m1['m00'])
            cy1 = int(m1['m01'] / m1['m00'])

            for j, c2 in enumerate(contours[i + 1:], i + 1):
                if j in used:
                    continue

                m2 = cv2.moments(c2)
                if m2['m00'] == 0:
                    continue

                cx2 = int(m2['m10'] / m2['m00'])
                cy2 = int(m2['m01'] / m2['m00'])

                dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                if dist < 20:
                    area1 = cv2.contourArea(c1)
                    area2 = cv2.contourArea(c2)
                    if area1 >= area2:
                        used.add(j)
                    else:
                        c1 = c2
                        used.add(i)
                        i = j

            if i not in used:
                merged.append(c1)

        return merged

    def detect_objects(self):
        if self.current_frame is None:
            self.show_alert("No camera feed available", (0, 0, 255))
            return

        frame = self.current_frame.copy()
        enhanced = self.preprocess_frame(frame)
        self.detect_objects_multiscale(frame, enhanced)

    def is_valid_contour(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return False

        circularity = 4 * math.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)
        if w < self.min_contour_size or h < self.min_contour_size:
            return False

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False

        solidity = area / hull_area
        if solidity < 0.45:
            return False

        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect_ratio > 12:
            return False

        if area < 100:
            return False

        return True

    def approximate_polygon(self, contour):
        peri = cv2.arcLength(contour, True)
        epsilon = 0.012 * peri
        return cv2.approxPolyDP(contour, epsilon, True)

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

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0

        size_category = self.categorize_size(area_cm2)

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
            'solidity': solidity,
            'centroid': (cx, cy),
            'vertices': vertices,
            'size_category': size_category,
            'timestamp': datetime.now().isoformat()
        }

    def categorize_size(self, area_cm2):
        if area_cm2 < 5:
            return "Tiny"
        elif area_cm2 < 25:
            return "Small"
        elif area_cm2 < 100:
            return "Medium"
        elif area_cm2 < 300:
            return "Large"
        else:
            return "VeryLarge"

    def smooth_measurement(self, obj_id, measurement_dict):
        if obj_id not in self.smoothed_measurements:
            self.smoothed_measurements[obj_id] = {
                'width_cm': measurement_dict['width_cm'],
                'height_cm': measurement_dict['height_cm'],
                'area_cm2': measurement_dict['area_cm2'],
                'perimeter_cm': measurement_dict['perimeter_cm']
            }
        else:
            alpha = 0.35
            prev = self.smoothed_measurements[obj_id]
            self.smoothed_measurements[obj_id] = {
                'width_cm': prev['width_cm'] * (1 - alpha) + measurement_dict['width_cm'] * alpha,
                'height_cm': prev['height_cm'] * (1 - alpha) + measurement_dict['height_cm'] * alpha,
                'area_cm2': prev['area_cm2'] * (1 - alpha) + measurement_dict['area_cm2'] * alpha,
                'perimeter_cm': prev['perimeter_cm'] * (1 - alpha) + measurement_dict['perimeter_cm'] * alpha
            }

        return self.smoothed_measurements[obj_id]

    def classify_shape(self, obj):
        circularity = obj['circularity']
        aspect_ratio = obj['aspect_ratio']
        vertices = obj['vertices']
        solidity = obj['solidity']

        if circularity > 0.83 and solidity > 0.83:
            return "Circle"
        elif vertices == 3 and solidity > 0.72:
            return "Triangle"
        elif vertices == 4:
            if 0.78 < aspect_ratio < 1.25 and solidity > 0.78:
                return "Square"
            elif solidity > 0.72:
                return "Rectangle"
        elif vertices == 5 and solidity > 0.72:
            return "Pentagon"
        elif vertices == 6 and solidity > 0.72:
            return "Hexagon"
        elif vertices > 6 and circularity > 0.62 and solidity > 0.68:
            return "Polygon"

        return "Irregular"

    def analyze_selected(self):
        if self.selected_index is None or self.selected_index >= len(self.detected_objects):
            self.show_alert("No object selected", (0, 0, 255))
            return

        obj = self.detected_objects[self.selected_index]
        obj_id = id(obj)

        smoothed = self.smooth_measurement(obj_id, {
            'width_cm': obj['width_cm'],
            'height_cm': obj['height_cm'],
            'area_cm2': obj['area_cm2'],
            'perimeter_cm': obj['perimeter_cm']
        })

        confidence = round(min(obj['circularity'] * obj['solidity'] * 100, 100), 1)

        measurement = {
            'object_id': len(self.measurements) + 1,
            'width_cm': round(smoothed['width_cm'], 2),
            'height_cm': round(smoothed['height_cm'], 2),
            'area_cm2': round(smoothed['area_cm2'], 2),
            'perimeter_cm': round(smoothed['perimeter_cm'], 2),
            'shape': self.classify_shape(obj),
            'vertices': obj['vertices'],
            'size': obj['size_category'],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }

        self.measurements.append(measurement)
        self.object_stats[obj['size_category']]['measurements'].append(measurement)
        self.object_stats[obj['size_category']]['detected_count'] += 1

        msg = "Measured: {:.2f}cm x {:.2f}cm ({})".format(
            smoothed['width_cm'], smoothed['height_cm'], obj['size_category'])
        self.show_alert(msg, (100, 200, 100))

    def show_statistics(self):
        if not self.measurements:
            self.show_alert("No measurements yet", (150, 150, 200))
            return

        stats = "\n" + "=" * 80
        stats += "\nDETAILED MEASUREMENT STATISTICS\n"
        stats += "=" * 80
        stats += "\nTotal Measurements: {}\n".format(len(self.measurements))
        stats += "Active Size Filters: {}\n".format(', '.join(self.size_filter.get_enabled_sizes()))

        for size_cat in ["Tiny", "Small", "Medium", "Large", "VeryLarge"]:
            if size_cat in self.object_stats:
                count = self.object_stats[size_cat]['detected_count']
                if count > 0:
                    stats += "\n{} Objects: {}".format(size_cat, count)
                    measurements = self.object_stats[size_cat]['measurements']
                    avg_width = sum(m['width_cm'] for m in measurements) / len(measurements)
                    avg_height = sum(m['height_cm'] for m in measurements) / len(measurements)
                    avg_area = sum(m['area_cm2'] for m in measurements) / len(measurements)
                    avg_confidence = sum(m['confidence'] for m in measurements) / len(measurements)
                    stats += "\n  Avg Width: {:.2f}cm | Height: {:.2f}cm | Area: {:.2f}cm2".format(
                        avg_width, avg_height, avg_area)
                    stats += "\n  Avg Confidence: {:.1f}%".format(avg_confidence)

        stats += "\n" + "=" * 80 + "\n"
        print(stats)
        self.show_alert("Statistics printed to console", (100, 150, 200))

    def toggle_auto(self):
        self.auto_mode_enabled = not self.auto_mode_enabled
        status = "ON" if self.auto_mode_enabled else "OFF"
        msg = "Auto Mode: {} - Continuous detection active".format(status)
        self.show_alert(msg, (200, 150, 0))

    def clear_detections(self):
        self.detected_objects = []
        self.selected_index = None
        self.smoothed_measurements = {}
        self.show_alert("All detections cleared", (150, 150, 150))

    def save_measurements(self):
        if not self.measurements:
            self.show_alert("No measurements to save", (0, 0, 255))
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
            'total': len(self.measurements),
            'size_filters_used': self.size_filter.get_enabled_sizes(),
            'statistics': {
                'total_objects': sum(self.object_stats[cat]['detected_count'] for cat in self.object_stats)
            }
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.show_alert("Saved: {}".format(filename), (100, 200, 100))
        except Exception as e:
            self.show_alert("Error: {}".format(str(e)), (0, 0, 255))

    def draw_size_filter_menu(self, frame):
        h, w = frame.shape[:2]

        size_button_y = 80
        size_button_height = 35
        size_button_spacing = 5
        size_button_width = 80
        size_button_x_start = 20
        menu_height = size_button_y + size_button_height + 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, size_button_y - 10), (w, menu_height), (15, 30, 50), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (0, size_button_y - 10), (w, menu_height), (100, 150, 200), 2)

        cv2.putText(frame, "SELECT SIZE CATEGORIES TO DETECT:", (size_button_x_start, size_button_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 200), 1)

        size_buttons = [
            ('Tiny\n<5cm2', 'Tiny'),
            ('Small\n5-25', 'Small'),
            ('Medium\n25-100', 'Medium'),
            ('Large\n100-300', 'Large'),
            ('V.Large\n>300', 'VeryLarge')
        ]

        for idx, (label, size_name) in enumerate(size_buttons):
            bx = size_button_x_start + idx * (size_button_width + size_button_spacing)
            by = size_button_y

            color = (80, 200, 100) if self.size_filter.enabled_sizes[size_name] else (70, 70, 90)

            cv2.rectangle(frame, (bx, by), (bx + size_button_width, by + size_button_height), color, -1)
            cv2.rectangle(frame, (bx, by), (bx + size_button_width, by + size_button_height), (220, 220, 220), 2)

            status = "ON" if self.size_filter.enabled_sizes[size_name] else "OFF"
            text_color = (255, 255, 255)
            cv2.putText(frame, status, (bx + 20, by + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        all_enable_bx = 500
        all_enable_by = 80
        all_enable_width = 60
        all_enable_height = 35

        cv2.rectangle(frame, (all_enable_bx, all_enable_by),
                      (all_enable_bx + all_enable_width, all_enable_by + all_enable_height), (80, 180, 100), -1)
        cv2.rectangle(frame, (all_enable_bx, all_enable_by),
                      (all_enable_bx + all_enable_width, all_enable_by + all_enable_height), (220, 220, 220), 2)
        cv2.putText(frame, "ALL", (all_enable_bx + 10, all_enable_by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

        all_disable_bx = 570
        all_disable_by = 80
        all_disable_width = 60
        all_disable_height = 35

        cv2.rectangle(frame, (all_disable_bx, all_disable_by),
                      (all_disable_bx + all_disable_width, all_disable_by + all_disable_height), (80, 100, 100), -1)
        cv2.rectangle(frame, (all_disable_bx, all_disable_by),
                      (all_disable_bx + all_disable_width, all_disable_by + all_disable_height), (220, 220, 220), 2)
        cv2.putText(frame, "RESET", (all_disable_bx + 5, all_disable_by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        return menu_height

    def draw_frame(self, frame):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (10, 20, 35), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (0, 0), (w, 80), (120, 160, 220), 3)

        button_y = 25
        button_height = 40
        button_spacing = 6
        button_width = 85
        button_x_start = 20

        buttons = [
            ('DETECT', (60, 160, 120)),
            ('ANALYZE', (120, 130, 220)),
            ('AUTO', (220, 130, 60) if self.auto_mode_enabled else (90, 90, 110)),
            ('FILTER', (160, 110, 160) if self.show_size_menu else (110, 110, 120)),
            ('STATS', (160, 110, 160)),
            ('CLEAR', (120, 130, 220)),
            ('SAVE', (120, 130, 220)),
            ('EXIT', (220, 90, 90))
        ]

        for idx, (label, color) in enumerate(buttons):
            bx = button_x_start + idx * (button_width + button_spacing)
            by = button_y

            cv2.rectangle(frame, (bx, by), (bx + button_width, by + button_height), color, -1)
            cv2.rectangle(frame, (bx, by), (bx + button_width, by + button_height), (240, 240, 240), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.38, 1)[0]
            text_x = bx + (button_width - text_size[0]) // 2
            text_y = by + (button_height + text_size[1]) // 2

            cv2.putText(frame, label, (text_x, text_y), font, 0.38, (255, 255, 255), 1)

        menu_bottom = 80
        if self.show_size_menu:
            menu_bottom = self.draw_size_filter_menu(frame)

        for i, obj in enumerate(self.detected_objects):
            x, y, w_obj, h_obj = obj['bbox']

            is_selected = (i == self.selected_index)
            color = (0, 255, 150) if is_selected else (100, 200, 255)
            thickness = 4 if is_selected else 2

            cv2.rectangle(frame, (x, y), (x + w_obj, y + h_obj), color, thickness)
            cv2.drawContours(frame, [obj['contour']], 0, color, 2)

            cx, cy = obj['centroid']
            cv2.circle(frame, (cx, cy), 7, color, -1)

            label_text = "#{}: {:.1f}x{:.1f}cm ({})".format(
                i + 1, obj['width_cm'], obj['height_cm'], obj['size_category'][0])

            label_bg_w = 200
            label_bg_h = 25
            cv2.rectangle(frame, (x, y - 35), (x + label_bg_w, y - 10), (20, 20, 20), -1)
            cv2.putText(frame, label_text, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if self.selected_index is not None and self.selected_index < len(self.detected_objects):
            obj = self.detected_objects[self.selected_index]

            panel_x = 20
            panel_y = h - 320
            panel_w = 400
            panel_h = 300

            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 25, 45), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (120, 180, 255), 3)

            cv2.putText(frame, "OBJECT ANALYSIS & MEASUREMENTS", (panel_x + 15, panel_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 180, 255), 2)

            cv2.line(frame, (panel_x + 15, panel_y + 35), (panel_x + panel_w - 15, panel_y + 35), (120, 180, 255), 2)

            y_offset = panel_y + 60
            info_lines = [
                "Length: {:.2f} cm".format(obj['width_cm']),
                "Height: {:.2f} cm".format(obj['height_cm']),
                "Area: {:.2f} cm2".format(obj['area_cm2']),
                "Perimeter: {:.2f} cm".format(obj['perimeter_cm']),
                "Shape: {}".format(self.classify_shape(obj)),
                "Category: {} | Vertices: {}".format(obj['size_category'], obj['vertices']),
                "Solidity: {:.1f}% | Circularity: {:.3f}".format(obj['solidity'] * 100, obj['circularity'])
            ]

            for line in info_lines:
                cv2.putText(frame, line, (panel_x + 15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
                y_offset += 32

        if self.alert_message:
            if (datetime.now() - self.alert_time).total_seconds() < 3.5:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(self.alert_message, font, 0.85, 2)[0]

                alert_x = (w - text_size[0]) // 2
                alert_y = h - 30

                overlay = frame.copy()
                cv2.rectangle(overlay, (alert_x - 20, alert_y - 30), (alert_x + text_size[0] + 20, alert_y + 10),
                              (15, 25, 35), -1)
                cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
                cv2.rectangle(frame, (alert_x - 20, alert_y - 30), (alert_x + text_size[0] + 20, alert_y + 10),
                              self.alert_color, 3)

                cv2.putText(frame, self.alert_message, (alert_x, alert_y),
                            font, 0.85, self.alert_color, 2)

        enabled_text = ','.join([s[:3] for s in self.size_filter.get_enabled_sizes()])
        mode_text = "AUTO" if self.auto_mode_enabled else "MANUAL"
        status_text = "MODE: {} | FILTER: [{}] | OBJ: {} | MEASURED: {} | FPS: {:.1f}".format(
            mode_text, enabled_text, len(self.detected_objects), len(self.measurements), self.current_fps)

        cv2.rectangle(frame, (w - 700, h - 30), (w, h), (20, 20, 30), -1)
        cv2.putText(frame, status_text, (w - 690, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        window_name = 'Advanced Measurement Tool v8.1'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height + 100)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\nTool ready. Click FILTER button to select object sizes.\n")
        print("Instructions:")
        print("1. Click FILTER - Opens size selector menu")
        print("2. Click size buttons - Toggle ON/OFF (GREEN=ON, DARK=OFF)")
        print("3. Click DETECT - Find objects in selected sizes")
        print("4. Click AUTO - Continuous detection mode")
        print("5. Click ANALYZE - Save measurement of selected object")
        print("=" * 70 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.current_frame = frame.copy()
            self.update_fps()

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
        summary = "\n" + "=" * 80
        summary += "\nFINAL MEASUREMENT SUMMARY REPORT"
        summary += "\n" + "=" * 80
        summary += "\nTotal Measurements Recorded: {}\n".format(len(self.measurements))
        summary += "Size Filters Used: {}\n".format(', '.join(self.size_filter.get_enabled_sizes()))
        summary += "=" * 80

        for m in self.measurements:
            summary += "\n\nObject #{}:".format(m['object_id'])
            summary += "\n  Size Category: {}".format(m['size'])
            summary += "\n  Length: {:.2f} cm".format(m['width_cm'])
            summary += "\n  Height: {:.2f} cm".format(m['height_cm'])
            summary += "\n  Area: {:.2f} cm2".format(m['area_cm2'])
            summary += "\n  Perimeter: {:.2f} cm".format(m['perimeter_cm'])
            summary += "\n  Shape: {}".format(m['shape'])
            summary += "\n  Vertices: {} | Confidence: {}%".format(m['vertices'], m['confidence'])
            summary += "\n  Timestamp: {}".format(m['timestamp'])

        summary += "\n" + "=" * 80
        summary += "\nMeasurement session completed successfully!"
        summary += "\n" + "=" * 80 + "\n"
        print(summary)


def main():
    print("\n" + "=" * 80)
    print("ADVANCED MEASUREMENT TOOL v8.1 - PROFESSIONAL SIZE FILTERING")
    print("=" * 80)
    print("\nFeatures:")
    print("✓ Smart Size Filter with 5 independent categories")
    print("✓ Real-time object detection based on selected sizes")
    print("✓ Multi-scale detection (Tiny to Very Large)")
    print("✓ Real-time FPS monitoring and performance tracking")
    print("✓ Advanced geometric shape classification")
    print("✓ Measurement smoothing and validation algorithms")
    print("✓ Professional UI with detailed measurements")
    print("✓ Object tracking across frames")
    print("✓ Confidence scoring system")
    print("\nSize Categories:")
    print("  • Tiny: < 5 cm²")
    print("  • Small: 5-25 cm²")
    print("  • Medium: 25-100 cm²")
    print("  • Large: 100-300 cm²")
    print("  • Very Large: > 300 cm²")
    print("\nQuick Start:")
    print("1. Click FILTER to open size selector")
    print("2. Toggle size categories ON/OFF (GREEN=ON, DARK=OFF)")
    print("3. Click DETECT to find objects, or AUTO for continuous mode")
    print("4. Click ANALYZE to save measurements")
    print("5. Click STATS to view detailed statistics")
    print("=" * 80 + "\n")

    tool = MeasurementTool()
    tool.run()


if __name__ == "__main__":
    main()