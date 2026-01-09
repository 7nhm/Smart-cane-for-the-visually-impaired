import cv2
import numpy as np
import time
from collections import deque
import os

class AdvancedObstacleDetector:
    def __init__(self):
        print("Loading Advanced Obstacle Detector...")
        
        # ================== THAM SỐ CỐ ĐỊNH ==================
        self.min_obstacle_area = 800
        self.max_obstacle_area = 30000
        self.min_aspect_ratio = 0.25
        self.max_aspect_ratio = 3.5
        self.min_confidence = 0.65
        self.roi_y_start = 0.3
        self.roi_y_end = 0.9
        
        # ================== INITIALIZE DETECTORS ==================
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=200, 
            varThreshold=25, 
            detectShadows=False
        )
        
        self.object_cascade = None
        self._load_cascade()
        
        # ================== TRACKING ==================
        self.detection_history = deque(maxlen=5)
        self.obstacle_tracker = {}
        self.track_id_counter = 0
        self.obstacle_window = deque(maxlen=10)
        
        print("Advanced Obstacle Detector ready!")
    
    def _load_cascade(self):
        """Tải cascade classifier"""
        try:
            cascade_paths = [
                cv2.data.haarcascades + "haarcascade_fullbody.xml",
                cv2.data.haarcascades + "haarcascade_upperbody.xml",
                cv2.data.haarcascades + "haarcascade_lowerbody.xml"
            ]
            
            for cascade_path in cascade_paths:
                if os.path.exists(cascade_path):
                    self.object_cascade = cv2.CascadeClassifier(cascade_path)
                    if not self.object_cascade.empty():
                        print(f"Loaded: {os.path.basename(cascade_path)}")
                        return
            
            print("No cascade found")
            self.object_cascade = None
            
        except Exception as e:
            print(f"Error: {e}")
            self.object_cascade = None
    
    def _detect_using_cascade(self, frame):
        """Phát hiện bằng cascade"""
        if self.object_cascade is None or frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        objects = self.object_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        obstacles = []
        for (x, y, w, h) in objects:
            area = w * h
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                aspect_ratio = w / (h + 1e-6)
                
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    confidence = 0.85
                    
                    obstacles.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': confidence,
                        'method': 'cascade'
                    })
        
        return obstacles
    
    def _detect_using_background_subtraction(self, frame):
        """Phát hiện bằng background subtraction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        fgmask = self.fgbg.apply(gray, learningRate=0.003)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        _, fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                aspect_ratio = w / (h + 1e-6)
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0
                    
                    if solidity > 0.3:
                        area_confidence = min(1.0, area / 8000)
                        confidence = 0.7 * area_confidence + 0.3 * solidity
                        
                        obstacles.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'confidence': confidence,
                            'method': 'background_sub'
                        })
        
        return obstacles, fgmask
    
    def _detect_using_edge_density(self, frame):
        """Phát hiện bằng mật độ cạnh"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)
        
        _, edge_mask = cv2.threshold(magnitude, 40, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                roi_edges = edge_mask[y:y+h, x:x+w]
                edge_density = np.sum(roi_edges > 0) / (w * h + 1e-6)
                
                if 0.1 <= edge_density <= 0.8:
                    density_confidence = min(1.0, edge_density * 1.5)
                    area_confidence = min(1.0, area / 6000)
                    confidence = 0.5 * density_confidence + 0.5 * area_confidence
                    
                    obstacles.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': confidence,
                        'method': 'edge_density'
                    })
        
        return obstacles, edge_mask
    
    def _detect_using_motion(self, current_frame):
        """Phát hiện chuyển động"""
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return [], None
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        motion_mask = np.zeros_like(current_gray, dtype=np.uint8)
        motion_mask[mag > 1.5] = 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                roi_mag = mag[y:y+h, x:x+w]
                motion_intensity = np.mean(roi_mag) if roi_mag.size > 0 else 0
                
                motion_confidence = min(1.0, motion_intensity / 4.0)
                area_confidence = min(1.0, area / 8000)
                confidence = 0.6 * motion_confidence + 0.4 * area_confidence
                
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'method': 'motion'
                })
        
        self.prev_gray = current_gray
        
        return obstacles, motion_mask
    
    def _apply_roi_filter(self, obstacles, frame_height):
        """Lọc obstacles trong ROI"""
        filtered_obstacles = []
        
        for obstacle in obstacles:
            x, y, w, h = obstacle['bbox']
            center_y = y + h // 2
            
            roi_start = int(frame_height * self.roi_y_start)
            roi_end = int(frame_height * self.roi_y_end)
            
            if roi_start <= center_y <= roi_end:
                distance_score = ((center_y - roi_start) / (roi_end - roi_start))
                
                new_confidence = obstacle['confidence'] * (0.8 + 0.2 * distance_score)
                obstacle['confidence'] = min(new_confidence, 1.0)
                obstacle['distance_score'] = distance_score
                
                filtered_obstacles.append(obstacle)
        
        return filtered_obstacles
    
    def _temporal_filter(self, current_obstacles):
        """Lọc nhiễu thời gian"""
        self.obstacle_window.append(len(current_obstacles) > 0)
        
        if len(self.obstacle_window) == self.obstacle_window.maxlen:
            obstacle_count = sum(self.obstacle_window)
            if obstacle_count >= 6:
                return current_obstacles
            else:
                return []
        
        self.detection_history.append(current_obstacles)
        
        if len(self.detection_history) < 2:
            return current_obstacles
        
        consistent_obstacles = []
        
        for current_obs in current_obstacles:
            current_bbox = current_obs['bbox']
            current_center = (current_bbox[0] + current_bbox[2] // 2,
                            current_bbox[1] + current_bbox[3] // 2)
            
            appearance_count = 0
            
            for hist_obstacles in list(self.detection_history)[-2:]:
                found = False
                
                for hist_obs in hist_obstacles:
                    hist_bbox = hist_obs['bbox']
                    hist_center = (hist_bbox[0] + hist_bbox[2] // 2,
                                  hist_bbox[1] + hist_bbox[3] // 2)
                    
                    distance = np.sqrt((current_center[0] - hist_center[0])**2 + 
                                      (current_center[1] - hist_center[1])**2)
                    
                    if distance < 40:
                        found = True
                        break
                
                if found:
                    appearance_count += 1
            
            if appearance_count >= 2:
                consistency_confidence = min(1.0, appearance_count / 2 * 1.2)
                current_obs['confidence'] = min(1.0, current_obs['confidence'] * consistency_confidence)
                current_obs['consistency_score'] = consistency_confidence
                consistent_obstacles.append(current_obs)
        
        return consistent_obstacles
    
    def _update_obstacle_tracker(self, obstacles, frame_shape):
        """Cập nhật tracker"""
        h, w = frame_shape[:2]
        current_time = time.time()
        
        current_obstacles = []
        for obs in obstacles:
            x, y, w_box, h_box = obs['bbox']
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            current_obstacles.append({
                'center': (center_x, center_y),
                'bbox': obs['bbox'],
                'confidence': obs['confidence']
            })
        
        to_remove = []
        for track_id, track_info in list(self.obstacle_tracker.items()):
            if current_time - track_info['last_seen'] > 10:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.obstacle_tracker[track_id]
        
        matched_tracks = set()
        assigned_current = [False] * len(current_obstacles)
        
        if self.obstacle_tracker:
            for track_id, track_info in list(self.obstacle_tracker.items()):
                min_distance = float('inf')
                min_idx = -1
                
                for idx, obs in enumerate(current_obstacles):
                    if assigned_current[idx]:
                        continue
                    
                    distance = np.sqrt(
                        (obs['center'][0] - track_info['center'][0])**2 +
                        (obs['center'][1] - track_info['center'][1])**2
                    )
                    
                    if distance < 50 and distance < min_distance:
                        min_distance = distance
                        min_idx = idx
                
                if min_idx != -1:
                    self.obstacle_tracker[track_id] = {
                        'center': current_obstacles[min_idx]['center'],
                        'bbox': current_obstacles[min_idx]['bbox'],
                        'confidence': current_obstacles[min_idx]['confidence'],
                        'last_seen': current_time,
                        'age': track_info.get('age', 0) + 1
                    }
                    matched_tracks.add(track_id)
                    assigned_current[min_idx] = True
        
        for idx, obs in enumerate(current_obstacles):
            if not assigned_current[idx]:
                self.track_id_counter += 1
                self.obstacle_tracker[self.track_id_counter] = {
                    'center': obs['center'],
                    'bbox': obs['bbox'],
                    'confidence': obs['confidence'],
                    'last_seen': current_time,
                    'age': 1
                }
    
    def _get_stable_obstacles(self, min_age=2):
        """Lấy obstacles ổn định"""
        stable_obstacles = []
        
        for track_id, track_info in list(self.obstacle_tracker.items()):
            if track_info.get('age', 0) >= min_age:
                stable_obstacles.append({
                    'bbox': track_info['bbox'],
                    'confidence': track_info['confidence'],
                    'track_id': track_id,
                    'age': track_info['age']
                })
        
        return stable_obstacles
    
    def _non_maximum_suppression(self, obstacles, overlap_thresh=0.4):
        """Loại bỏ bounding box trùng lặp"""
        if len(obstacles) == 0:
            return []
        
        boxes = np.array([obs['bbox'] for obs in obstacles])
        confidences = np.array([obs['confidence'] for obs in obstacles])
        
        if boxes.size == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(confidences)[::-1]
        
        pick = []
        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            
            if len(idxs) == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[1:]]
            
            idxs = idxs[1:][overlap <= overlap_thresh]
        
        return [obstacles[i] for i in pick]
    
    def _calculate_danger_level(self, obstacles, frame_height, frame_width):
        """Tính mức độ nguy hiểm"""
        if not obstacles:
            return 0
        
        max_danger = 0
        
        for obs in obstacles:
            x, y, w, h = obs['bbox']
            center_y = y + h // 2
            center_x = x + w // 2
            
            roi_start = int(frame_height * self.roi_y_start)
            roi_end = int(frame_height * self.roi_y_end)
            
            if center_y < roi_start or center_y > roi_end:
                continue
            
            if roi_end != roi_start:
                vertical_danger = (center_y - roi_start) / (roi_end - roi_start)
            else:
                vertical_danger = 0.5
            
            horizontal_center = abs(center_x - frame_width / 2) / (frame_width / 2)
            horizontal_danger = 1.0 - horizontal_center
            
            bbox_area = w * h
            frame_area = frame_width * frame_height
            area_ratio = bbox_area / frame_area
            
            touches_edge = 0
            if x <= 5: touches_edge += 1
            if x + w >= frame_width - 5: touches_edge += 1
            if y <= 5: touches_edge += 1
            if y + h >= frame_height - 5: touches_edge += 1
            
            edge_penalty = 1.0 + (touches_edge * 0.2)
            
            conf_score = obs['confidence']
            
            danger_score = (
                0.5 * area_ratio * 3.0 +
                0.25 * vertical_danger +
                0.15 * conf_score +
                0.1 * horizontal_danger
            ) * edge_penalty
            
            if danger_score > 0.7:
                danger_level = 2
            elif danger_score > 0.4:
                danger_level = 1
            else:
                danger_level = 0
            
            max_danger = max(max_danger, danger_level)
        
        return max_danger
    
    def detect_obstacles(self, frame):
        """Phát hiện vật cản chính"""
        if frame is None or frame.size == 0:
            return False, [], 0.0, 0
        
        h, w = frame.shape[:2]
        
        all_obstacles = []
        
        bg_obstacles, bg_mask = self._detect_using_background_subtraction(frame)
        all_obstacles.extend(bg_obstacles)
        
        edge_obstacles, edge_mask = self._detect_using_edge_density(frame)
        all_obstacles.extend(edge_obstacles)
        
        motion_obstacles, motion_mask = self._detect_using_motion(frame)
        all_obstacles.extend(motion_obstacles)
        
        cascade_obstacles = self._detect_using_cascade(frame)
        all_obstacles.extend(cascade_obstacles)
        
        filtered_obstacles = self._apply_roi_filter(all_obstacles, h)
        
        final_obstacles = self._temporal_filter(filtered_obstacles)
        
        final_obstacles = [obs for obs in final_obstacles 
                          if obs['confidence'] >= self.min_confidence]
        
        self._update_obstacle_tracker(final_obstacles, frame.shape)
        
        stable_obstacles = self._get_stable_obstacles(min_age=2)
        
        if stable_obstacles:
            combined_obstacles = stable_obstacles
            for obs in final_obstacles:
                found = False
                for stable_obs in stable_obstacles:
                    x1, y1, w1, h1 = obs['bbox']
                    x2, y2, w2, h2 = stable_obs['bbox']
                    
                    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
                    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
                    
                    if dx > 0 and dy > 0:
                        found = True
                        break
                
                if not found:
                    combined_obstacles.append(obs)
        else:
            combined_obstacles = final_obstacles
        
        final_obstacles = self._non_maximum_suppression(combined_obstacles)
        
        has_obstacle = len(final_obstacles) > 0
        
        total_confidence = 0.0
        if has_obstacle:
            confidences = [obs['confidence'] for obs in final_obstacles]
            total_confidence = max(confidences) if confidences else 0.0
        
        danger_level = self._calculate_danger_level(final_obstacles, h, w)
        
        return has_obstacle, final_obstacles, total_confidence, danger_level
    
    def visualize_detection(self, frame, obstacles, danger_level):
        """Hiển thị kết quả"""
        if frame is None:
            return frame
        
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        
        roi_start = int(h * self.roi_y_start)
        roi_end = int(h * self.roi_y_end)
        cv2.rectangle(viz_frame, (0, roi_start), (w, roi_end), (255, 200, 0), 1)
        
        cv2.line(viz_frame, (w//2, 0), (w//2, h), (0, 200, 200), 1)
        
        for obs in obstacles:
            x, y, w_box, h_box = obs['bbox']
            confidence = obs['confidence']
            method = obs.get('method', 'unknown')
            track_id = obs.get('track_id', None)
            age = obs.get('age', 1)
            
            if age >= 5:
                color = (0, 0, 255)
                thickness = 3
                stability_text = f"Stable({age})"
            elif age >= 3:
                color = (0, 165, 255)
                thickness = 2
                stability_text = f"Good({age})"
            else:
                color = (0, 255, 255)
                thickness = 1
                stability_text = f"New({age})"
            
            cv2.rectangle(viz_frame, (x, y), (x + w_box, y + h_box), color, thickness)
            
            label = f"{method}: {confidence:.1f}"
            if track_id is not None:
                label = f"ID:{track_id} {label}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(viz_frame, (x, y - label_h - 15), 
                         (x + label_w, y), color, -1)
            
            cv2.putText(viz_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(viz_frame, stability_text, (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            cv2.circle(viz_frame, (center_x, center_y), 3, color, -1)
        
        if danger_level == 2:
            status_color = (0, 0, 255)
            status_text = "HIGH DANGER!"
        elif danger_level == 1:
            status_color = (0, 165, 255)
            status_text = "MEDIUM DANGER"
        else:
            status_color = (0, 255, 255)
            status_text = "LOW DANGER"
        
        cv2.putText(viz_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        count_text = f"Obstacles: {len(obstacles)}"
        cv2.putText(viz_frame, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        track_count = len(self.obstacle_tracker)
        track_text = f"Tracking: {track_count} objects"
        cv2.putText(viz_frame, track_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        return viz_frame
    def _detect_using_edge_density(self, frame):
        """Phát hiện vật cản bằng mật độ cạnh (tốt cho vật tĩnh)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ để giảm noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Tính gradient
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính magnitude gradient
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Chuẩn hóa
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)
        
        # Threshold để lấy vùng có nhiều cạnh
        _, edge_mask = cv2.threshold(magnitude, 40, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Tính mật độ cạnh trong vùng
                roi_edges = edge_mask[y:y+h, x:x+w]
                edge_density = np.sum(roi_edges > 0) / (w * h + 1e-6)
                
                # Vật cản thường có mật độ cạnh trung bình đến cao
                if 0.1 <= edge_density <= 0.8:
                    # Confidence dựa trên mật độ cạnh và diện tích
                    density_confidence = min(1.0, edge_density * 1.5)
                    area_confidence = min(1.0, area / 6000)
                    confidence = 0.5 * density_confidence + 0.5 * area_confidence
                    
                    obstacles.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'edge_density': edge_density,
                        'confidence': confidence,
                        'method': 'edge_density'
                    })
        
        return obstacles, edge_mask
    
    def _detect_using_motion(self, current_frame):
        """Phát hiện chuyển động giữa các frame"""
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return [], None
        
        # Chuyển sang grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Tính optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Tính magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Tạo mask từ magnitude
        motion_mask = np.zeros_like(current_gray, dtype=np.uint8)
        motion_mask[mag > 1.5] = 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area <= area <= self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Tính motion intensity trong vùng
                roi_mag = mag[y:y+h, x:x+w]
                motion_intensity = np.mean(roi_mag) if roi_mag.size > 0 else 0
                
                # Confidence dựa trên cường độ chuyển động
                motion_confidence = min(1.0, motion_intensity / 4.0)
                area_confidence = min(1.0, area / 8000)
                confidence = 0.6 * motion_confidence + 0.4 * area_confidence
                
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'motion_intensity': motion_intensity,
                    'confidence': confidence,
                    'method': 'motion'
                })
        
        # Lưu frame hiện tại cho lần sau
        self.prev_gray = current_gray
        
        return obstacles, motion_mask
    
    def _apply_roi_filter(self, obstacles, frame_height):
        """Chỉ giữ lại vật cản trong vùng phía trước (ROI)"""
        filtered_obstacles = []
        
        for obstacle in obstacles:
            x, y, w, h = obstacle['bbox']
            center_y = y + h // 2
            
            # Chỉ giữ vật cản trong ROI
            roi_start = int(frame_height * self.roi_y_start)
            roi_end = int(frame_height * self.roi_y_end)
            
            if roi_start <= center_y <= roi_end:
                # Tính khoảng cách (vật càng thấp càng gần)
                distance_score = ((center_y - roi_start) / (roi_end - roi_start))
                
                # Cập nhật confidence với distance score
                new_confidence = obstacle['confidence'] * (0.8 + 0.2 * distance_score)
                obstacle['confidence'] = min(new_confidence, 1.0)
                obstacle['distance_score'] = distance_score
                
                filtered_obstacles.append(obstacle)
        
        return filtered_obstacles
    
    def _temporal_filter(self, current_obstacles):
        """Lọc nhiễu thời gian - chỉ giữ vật cản xuất hiện liên tiếp"""
        self.obstacle_window.append(len(current_obstacles) > 0)
        
        # Nếu có vật cản trong đa số các frame gần đây
        if len(self.obstacle_window) == self.obstacle_window.maxlen:
            obstacle_count = sum(self.obstacle_window)
            if obstacle_count >= 6:  # Ít nhất 6/10 frame có vật cản
                return current_obstacles
            else:
                return []
        
        # Cách cũ (giữ cho backward compatibility)
        self.detection_history.append(current_obstacles)
        
        if len(self.detection_history) < 2:
            return current_obstacles
        
        consistent_obstacles = []
        
        for current_obs in current_obstacles:
            current_bbox = current_obs['bbox']
            current_center = (current_bbox[0] + current_bbox[2] // 2,
                            current_bbox[1] + current_bbox[3] // 2)
            
            # Đếm số lần xuất hiện trong history
            appearance_count = 0
            
            for hist_obstacles in list(self.detection_history)[-2:]:
                found_in_history = False
                
                for hist_obs in hist_obstacles:
                    hist_bbox = hist_obs['bbox']
                    hist_center = (hist_bbox[0] + hist_bbox[2] // 2,
                                  hist_bbox[1] + hist_bbox[3] // 2)
                    
                    # Kiểm tra nếu vị trí gần nhau
                    distance = np.sqrt((current_center[0] - hist_center[0])**2 + 
                                      (current_center[1] - hist_center[1])**2)
                    
                    if distance < 40:
                        found_in_history = True
                        break
                
                if found_in_history:
                    appearance_count += 1
            
            # Nếu xuất hiện đủ số frame liên tiếp
            if appearance_count >= 2:
                consistency_confidence = min(1.0, appearance_count / 2 * 1.2)
                current_obs['confidence'] = min(1.0, current_obs['confidence'] * consistency_confidence)
                current_obs['consistency_score'] = consistency_confidence
                consistent_obstacles.append(current_obs)
        
        return consistent_obstacles
    
    def detect_obstacles(self, frame):
        """
        Main detection function - kết hợp tất cả phương pháp
        Trả về:
            - has_obstacle: bool (có vật cản không)
            - obstacles: list các vật cản đã lọc
            - confidence: độ tin cậy tổng
            - danger_level: mức độ nguy hiểm (0-2)
        """
        if frame is None or frame.size == 0:
            return False, [], 0.0, 0
        
        h, w = frame.shape[:2]
        
        # Áp dụng tất cả phương pháp
        all_obstacles = []
        
        # 1. Background subtraction
        bg_obstacles, bg_mask = self._detect_using_background_subtraction(frame)
        all_obstacles.extend(bg_obstacles)
        
        # 2. Edge density
        edge_obstacles, edge_mask = self._detect_using_edge_density(frame)
        all_obstacles.extend(edge_obstacles)
        
        # 3. Motion detection
        motion_obstacles, motion_mask = self._detect_using_motion(frame)
        all_obstacles.extend(motion_obstacles)
        
        # 4. Cascade detection (nếu có)
        cascade_obstacles = self._detect_using_cascade(frame)
        all_obstacles.extend(cascade_obstacles)
        
        # 5. ROI filtering (chỉ giữ vật trong vùng phía trước)
        filtered_obstacles = self._apply_roi_filter(all_obstacles, h)
        
        # 6. Temporal filtering (lọc nhiễu thời gian)
        final_obstacles = self._temporal_filter(filtered_obstacles)
        
        # 7. Confidence filtering
        final_obstacles = [obs for obs in final_obstacles 
                          if obs['confidence'] >= self.min_confidence]
        
        # Cập nhật tracker
        self._update_obstacle_tracker(final_obstacles, frame.shape)
        
        # Lấy obstacles ổn định từ tracker
        stable_obstacles = self._get_stable_obstacles(min_age=2)
        
        # Kết hợp obstacles hiện tại và stable obstacles
        if stable_obstacles:
            # Ưu tiên stable obstacles
            combined_obstacles = stable_obstacles
            
            # Thêm obstacles hiện tại nếu chưa có trong stable
            for obs in final_obstacles:
                # Kiểm tra xem obstacle đã có trong stable chưa
                found = False
                for stable_obs in stable_obstacles:
                    x1, y1, w1, h1 = obs['bbox']
                    x2, y2, w2, h2 = stable_obs['bbox']
                    
                    # Tính overlap
                    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
                    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
                    
                    if dx > 0 and dy > 0:
                        found = True
                        break
                
                if not found:
                    combined_obstacles.append(obs)
        else:
            combined_obstacles = final_obstacles
        
        # Non-maximum suppression (tránh trùng lặp)
        final_obstacles = self._non_maximum_suppression(combined_obstacles)
        
        # Xác định kết quả
        has_obstacle = len(final_obstacles) > 0
        
        # Tính confidence tổng
        total_confidence = 0.0
        if has_obstacle:
            confidences = [obs['confidence'] for obs in final_obstacles]
            total_confidence = max(confidences) if confidences else 0.0
        
        # Tính danger level
        danger_level = self._calculate_danger_level(final_obstacles, h, w)
        
        return has_obstacle, final_obstacles, total_confidence, danger_level
    
    def _non_maximum_suppression(self, obstacles, overlap_thresh=0.4):
        """Loại bỏ các bounding box trùng lặp"""
        if len(obstacles) == 0:
            return []
        
        # Lấy bounding boxes
        boxes = np.array([obs['bbox'] for obs in obstacles])
        confidences = np.array([obs['confidence'] for obs in obstacles])
        
        if boxes.size == 0:
            return []
        
        # Chuyển đổi sang định dạng (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Tính diện tích
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sắp xếp theo confidence
        idxs = np.argsort(confidences)[::-1]
        
        pick = []
        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            
            if len(idxs) == 1:
                break
            
            # Tìm overlap
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[1:]]
            
            # Loại bỏ các box có overlap lớn
            idxs = idxs[1:][overlap <= overlap_thresh]
        
        return [obstacles[i] for i in pick]
    
    def _calculate_danger_level(self, obstacles, frame_height, frame_width):
        """Tính mức độ nguy hiểm dựa trên vị trí và kích thước"""
        if not obstacles:
            return 0
        
        max_danger = 0
        
        for obs in obstacles:
            x, y, w, h = obs['bbox']
            center_y = y + h // 2
            center_x = x + w // 2
            
            # 1. Vị trí theo chiều dọc (Y-axis)
            roi_start = int(frame_height * self.roi_y_start)
            roi_end = int(frame_height * self.roi_y_end)
            
            if center_y < roi_start or center_y > roi_end:
                continue
            
            # Vật càng ở dưới thấp càng nguy hiểm
            if roi_end != roi_start:
                vertical_danger = (center_y - roi_start) / (roi_end - roi_start)
            else:
                vertical_danger = 0.5
            
            # 2. Vị trí theo chiều ngang (X-axis)
            horizontal_center = abs(center_x - frame_width / 2) / (frame_width / 2)
            horizontal_danger = 1.0 - horizontal_center
            
            # 3. Kích thước vật cản
            bbox_area = w * h
            frame_area = frame_width * frame_height
            area_ratio = bbox_area / frame_area
            
            # 4. Phát hiện vật TRÀN KHUNG HÌNH
            touches_edge = 0
            if x <= 5: touches_edge += 1
            if x + w >= frame_width - 5: touches_edge += 1
            if y <= 5: touches_edge += 1
            if y + h >= frame_height - 5: touches_edge += 1
            
            edge_penalty = 1.0 + (touches_edge * 0.2)
            
            # 5. Confidence score
            conf_score = obs['confidence']
            
            # Tính tổng danger score
            danger_score = (
                0.5 * area_ratio * 3.0 +
                0.25 * vertical_danger +
                0.15 * conf_score +
                0.1 * horizontal_danger
            ) * edge_penalty
            
            # Chuyển sang danger level
            if danger_score > 0.7:
                danger_level = 2  # RẤT NGUY HIỂM
            elif danger_score > 0.4:
                danger_level = 1  # NGUY HIỂM
            else:
                danger_level = 0  # CẢNH BÁO
            
            max_danger = max(max_danger, danger_level)
        
        return max_danger
    
    def visualize_detection(self, frame, obstacles, danger_level):
        """Hiển thị kết quả phát hiện vật cản"""
        if frame is None:
            return frame
        
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        
        # Vẽ ROI zone
        roi_start = int(h * self.roi_y_start)
        roi_end = int(h * self.roi_y_end)
        cv2.rectangle(viz_frame, (0, roi_start), (w, roi_end), (255, 255, 0), 1)
        
        # Vẽ center line
        cv2.line(viz_frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)
        
        # Vẽ obstacles
        for obs in obstacles:
            x, y, w_bbox, h_bbox = obs['bbox']
            confidence = obs['confidence']
            method = obs.get('method', 'unknown')
            
            # Thêm track_id nếu có
            track_id = obs.get('track_id', None)
            age = obs.get('age', 1)
            
            # Màu sắc dựa trên độ ổn định
            if age >= 5:  # Xuất hiện ít nhất 5 frame
                color = (0, 0, 255)  # Đỏ - rất ổn định
                thickness = 3
                stability_text = f"Stable({age})"
            elif age >= 3:
                color = (0, 165, 255)  # Cam - khá ổn định
                thickness = 2
                stability_text = f"Good({age})"
            else:
                color = (0, 255, 255)  # Vàng - mới xuất hiện
                thickness = 1
                stability_text = f"New({age})"
            
            # Vẽ bounding box
            cv2.rectangle(viz_frame, (x, y), (x + w_bbox, y + h_bbox), color, thickness)
            
            # Vẽ label
            label = f"{method}: {confidence:.1f}"
            if track_id is not None:
                label = f"ID:{track_id} {label}"
            
            # Tính kích thước text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Vẽ background cho label
            cv2.rectangle(viz_frame, (x, y - label_h - 15), 
                         (x + label_w, y), color, -1)
            
            # Vẽ text
            cv2.putText(viz_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(viz_frame, stability_text, (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Vẽ center point
            center_x = x + w_bbox // 2
            center_y = y + h_bbox // 2
            cv2.circle(viz_frame, (center_x, center_y), 3, color, -1)
        
        # Hiển thị danger level
        if danger_level == 2:
            status_color = (0, 0, 255)  # Đỏ
            status_text = "HIGH DANGER!"
        elif danger_level == 1:
            status_color = (0, 165, 255)  # Cam
            status_text = "MEDIUM DANGER"
        else:
            status_color = (0, 255, 255)  # Vàng
            status_text = "LOW DANGER"
        
        cv2.putText(viz_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Hiển thị số lượng vật cản
        count_text = f"Obstacles: {len(obstacles)}"
        cv2.putText(viz_frame, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Hiển thị thông tin tracker
        track_count = len(self.obstacle_tracker)
        track_text = f"Tracking: {track_count} objects"
        cv2.putText(viz_frame, track_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        return viz_frame


# Global instance
advanced_detector = AdvancedObstacleDetector()