"""
================================================================================
MULTI-CAMERA PERSON DETECTION, TRACKING & RE-IDENTIFICATION SYSTEM
================================================================================
Features:
  - YOLOv8 person detection
  - Kalman Filter + Hungarian Algorithm tracking
  - Person Re-Identification (Re-ID) with 128-dim embeddings
  - Cross-camera ID synchronization
  - Per-person video recording & FFmpeg merging
  - FastAPI backend with MJPEG streaming
================================================================================
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import time
import json
import threading
import logging
from logging.handlers import RotatingFileHandler
import uuid
import tempfile
import subprocess
import hashlib
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# pyright: reportMissingImports=false
try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ------------------------------------------------------------
# Configuration (from environment or defaults)
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Camera configs - support multiple cameras
CAMERA_CONFIGS = json.loads(os.getenv('CAMERA_CONFIGS', '[]'))
if not CAMERA_CONFIGS:
    CAMERA_CONFIGS = [
        {"id": "cam1", "url": os.getenv('RTSP_URL_1', 'rtsp://nirbhay:1234567890@192.168.1.68:1945'), "enabled": True},
        {"id": "cam2", "url": os.getenv('RTSP_URL_2', 'rtsp://nirbhay:1234567890@192.168.1.69:1945'), "enabled": False},
    ]

FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '640'))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '480'))
FPS = int(os.getenv('FPS', '15'))

# Detection Configuration
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', '0.4'))

# Tracking Configuration
TRACKING_MAX_DISTANCE = int(os.getenv('TRACKING_MAX_DISTANCE', '100'))
TRACKING_MAX_FRAMES_LOST = int(os.getenv('TRACKING_MAX_FRAMES_LOST', '30'))
MIN_HITS_TO_CONFIRM = int(os.getenv('MIN_HITS_TO_CONFIRM', '3'))

# Re-ID Configuration
REID_EMBEDDING_DIM = int(os.getenv('REID_EMBEDDING_DIM', '128'))
REID_SIMILARITY_THRESHOLD = float(os.getenv('REID_SIMILARITY_THRESHOLD', '0.7'))

# Recording Configuration
RECORDING_ENABLED = os.getenv('RECORDING_ENABLED', 'true').lower() == 'true'
RECORDING_DIR = os.getenv('RECORDING_DIR', 'recordings')
MERGED_DIR = os.getenv('MERGED_DIR', 'merged_videos')
MAX_RECORDING_MINUTES = int(os.getenv('MAX_RECORDING_MINUTES', '60'))

# Model & Directories paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.abspath(os.path.join(MODEL_DIR, '..', 'client'))
CLIENT_INDEX = os.path.join(CLIENT_DIR, 'index.html')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(os.path.join(MODEL_DIR, 'app.log'), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs(os.path.join(MODEL_DIR, RECORDING_DIR), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, MERGED_DIR), exist_ok=True)


# ============================================================
# KALMAN FILTER TRACKER
# ============================================================
class KalmanTracker:
    """
    4-state Kalman Filter: [cx, cy, vx, vy]
    Predicts and smooths person centroid motion
    """
    
    def __init__(self, dt: float = 1.0/30.0):
        self.dt = dt
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.R = np.eye(2, dtype=np.float32) * 10.0
        
        # State covariance
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        self.state = None  # [cx, cy, vx, vy]
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
    def init(self, centroid: Tuple[float, float]):
        """Initialize with first detection"""
        self.state = np.array([centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        
    def predict(self) -> Optional[np.ndarray]:
        """Predict next state"""
        if self.state is None:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.state[:2].copy()
        
    def update(self, centroid: Tuple[float, float]):
        """Update with measurement"""
        if self.state is None:
            self.init(centroid)
            return
            
        z = np.array(centroid, dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance (Joseph form for stability)
        I_KH = np.eye(4, dtype=np.float32) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.hits += 1
        self.time_since_update = 0
        
    def get_position(self) -> Optional[np.ndarray]:
        """Get current position estimate"""
        if self.state is None:
            return None
        return self.state[:2].copy()
        
    def get_velocity(self) -> Optional[np.ndarray]:
        """Get velocity estimate"""
        if self.state is None:
            return None
        return self.state[2:].copy()


# ============================================================
# PERSON RE-IDENTIFICATION (Re-ID)
# ============================================================
class PersonReID:
    """
    Extracts embedding vector from person crop
    Uses lightweight CNN feature extractor
    """
    
    def __init__(self, embedding_dim: int = 128, device: str = 'cpu'):
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = None
        
        self.transform = self._build_transforms()
        self._load_model()
        
    def _build_transforms(self):
        """Build image preprocessing pipeline"""
        try:
            import torchvision.transforms as T
            return T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            return None
            
    def _load_model(self):
        """Load Re-ID model"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, Re-ID disabled")
            return

        try:
            class SimpleReID(nn.Module):
                def __init__(self, dim: int):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                    )
                    self.fc = nn.Linear(128, dim)
                    
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return nn.functional.normalize(x, p=2, dim=1)
            
            self.model = SimpleReID(self.embedding_dim).to(self.device)
            self.model.eval()
            logger.info("Re-ID feature extractor initialized")
        except Exception as e:
            logger.warning(f"Re-ID initialization warning: {e}")
            
    def extract(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from person crop (BGR image)
        """
        if self.model is None or person_crop is None or person_crop.size == 0:
            return None
            
        try:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            if self.transform is None:
                rgb_resized = cv2.resize(rgb, (128, 256))
                rgb_norm = rgb_resized.astype(np.float32) / 255.0
                rgb_norm = (rgb_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            else:
                from PIL import Image
                pil_img = Image.fromarray(rgb)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(tensor)
                
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Re-ID extraction error: {e}")
            return None


# ============================================================
# CROSS-CAMERA ID MANAGER
# ============================================================
class CrossCameraIDManager:
    """
    Manages global person IDs across multiple cameras
    Uses Re-ID embeddings for matching
    """
    
    def __init__(self, similarity_threshold: float = 0.7, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        
        # Global DB: {global_id: {'embedding': np.array, 'last_seen': datetime, 'cameras': set}}
        self.global_db: Dict[str, Dict] = {}
        self.next_global_id = 1
        self.lock = threading.Lock()
        
    def match_or_create(self, local_embedding: Optional[np.ndarray], camera_id: str) -> str:
        """Match to global ID or create new"""
        with self.lock:
            if local_embedding is None:
                return self._create_new(camera_id)
            
            best_match = None
            best_sim = 0.0
            
            for gid, data in self.global_db.items():
                sim = self._cosine_similarity(local_embedding, data['embedding'])
                if sim > best_sim and sim > self.similarity_threshold:
                    best_sim = sim
                    best_match = gid
            
            if best_match:
                # Update existing
                old_emb = self.global_db[best_match]['embedding']
                self.global_db[best_match]['embedding'] = 0.7 * old_emb + 0.3 * local_embedding
                self.global_db[best_match]['last_seen'] = datetime.now()
                self.global_db[best_match]['cameras'].add(camera_id)
                logger.debug(f"Cross-camera match: {best_match} (sim: {best_sim:.3f})")
                return best_match
            else:
                return self._create_new(camera_id, local_embedding)
    
    def _create_new(self, camera_id: str, embedding: Optional[np.ndarray] = None) -> str:
        """Create new global ID"""
        gid = f"GLOBAL-{self.next_global_id:04d}"
        self.next_global_id += 1
        
        self.global_db[gid] = {
            'embedding': embedding if embedding is not None else np.random.randn(REID_EMBEDDING_DIM),
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'cameras': {camera_id}
        }
        
        if len(self.global_db) > self.max_entries:
            self._cleanup()
            
        logger.info(f"New global ID: {gid}")
        return gid
    
    def _cosine_similarity(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Cosine similarity"""
        if a is None or b is None:
            return 0.0
        norm = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / (norm + 1e-8)) if norm > 0 else 0.0
    
    def _cleanup(self):
        """Remove oldest entries"""
        sorted_ids = sorted(self.global_db.keys(), key=lambda k: self.global_db[k]['last_seen'])
        to_remove = sorted_ids[:len(sorted_ids)//4]
        for gid in to_remove:
            del self.global_db[gid]
        logger.info(f"Cleaned up {len(to_remove)} old global IDs")
    
    def get_info(self, global_id: str) -> Optional[Dict]:
        """Get global ID info"""
        with self.lock:
            if global_id in self.global_db:
                data = self.global_db[global_id]
                return {
                    'global_id': global_id,
                    'first_seen': data['first_seen'].isoformat(),
                    'last_seen': data['last_seen'].isoformat(),
                    'cameras': list(data['cameras'])
                }
            return None
    
    def get_all_ids(self) -> List[str]:
        """Get all global IDs"""
        with self.lock:
            return list(self.global_db.keys())


# ============================================================
# IMPROVED PERSON TRACKER (Kalman + Hungarian)
# ============================================================
class ImprovedPersonTracker:
    """
    Advanced tracker with:
    - Kalman Filter for motion prediction
    - Hungarian Algorithm for optimal assignment
    - Re-ID embedding storage
    """
    
    def __init__(self, max_distance: int = 100, max_frames_lost: int = 30, min_hits: int = 3):
        self.next_id = 1
        self.tracks: Dict[str, Dict] = {}
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.min_hits = min_hits
        
        self.person_database: Dict[str, Dict] = {}
        self.embeddings: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
    def update(self, detections: List[List[int]], 
               embeddings: Optional[List[np.ndarray]] = None) -> List[Tuple[str, List[int]]]:
        """
        Update tracks with new detections
        
        Returns:
            List of (local_person_id, bbox)
        """
        with self.lock:
            # Predict all existing tracks
            predicted_positions = {}
            for pid, track in self.tracks.items():
                pred = track['kalman'].predict()
                predicted_positions[pid] = pred if pred is not None else np.array(track['centroid'])
            
            # Calculate detection centroids
            det_centroids = []
            for det in detections:
                x, y, w, h = det
                det_centroids.append((x + w//2, y + h//2))
            
            # Hungarian matching
            matched_pairs = self._hungarian_match(
                list(predicted_positions.values()),
                det_centroids
            )
            
            matched_tracks = set()
            matched_dets = set()
            result = []
            
            # Process matches
            track_id_list = list(self.tracks.keys())
            for track_idx, det_idx in matched_pairs:
                if track_idx >= len(track_id_list) or det_idx >= len(detections):
                    continue
                    
                pid = track_id_list[track_idx]
                dist = float(np.linalg.norm(predicted_positions[pid] - np.array(det_centroids[det_idx])))
                
                if dist < self.max_distance:
                    self._update_track(pid, detections[det_idx], det_centroids[det_idx])
                    if embeddings and det_idx < len(embeddings):
                        self._store_embedding(pid, embeddings[det_idx])
                    result.append((pid, detections[det_idx]))
                    matched_tracks.add(pid)
                    matched_dets.add(det_idx)
            
            # New tracks for unmatched detections
            for det_idx, det in enumerate(detections):
                if det_idx not in matched_dets:
                    pid = self._create_person(det, det_centroids[det_idx])
                    if embeddings and det_idx < len(embeddings):
                        self._store_embedding(pid, embeddings[det_idx])
                    result.append((pid, det))
            
            # Mark unmatched tracks as lost
            for pid in list(self.tracks.keys()):
                if pid not in matched_tracks:
                    self.tracks[pid]['frames_lost'] += 1
                    if self.tracks[pid]['frames_lost'] <= self.max_frames_lost:
                        pred_pos = self.tracks[pid]['kalman'].get_position()
                        if pred_pos is not None:
                            old_bbox = self.tracks[pid]['bbox']
                            w, h = old_bbox[2], old_bbox[3]
                            new_x = int(pred_pos[0] - w//2)
                            new_y = int(pred_pos[1] - h//2)
                            self.tracks[pid]['bbox'] = [new_x, new_y, w, h]
                            result.append((pid, self.tracks[pid]['bbox']))
                    else:
                        logger.info(f"Person lost: {pid}")
                        del self.tracks[pid]
            
            return result
    
    def _hungarian_match(self, track_positions: List[np.ndarray], 
                         det_centroids: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimal assignment using Hungarian Algorithm"""
        if not track_positions or not det_centroids:
            return []
        
        cost_matrix = np.zeros((len(track_positions), len(det_centroids)))
        for i, tp in enumerate(track_positions):
            for j, dc in enumerate(det_centroids):
                cost_matrix[i, j] = np.linalg.norm(tp - np.array(dc))
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        return list(zip(track_indices, det_indices))
    
    def _create_person(self, bbox: List[int], centroid: Tuple[int, int]) -> str:
        """Create new track"""
        person_id = f"PERSON-{self.next_id:04d}"
        self.next_id += 1
        
        kalman = KalmanTracker(dt=1.0/FPS)
        kalman.init(centroid)
        
        self.tracks[person_id] = {
            'kalman': kalman,
            'bbox': bbox,
            'centroid': centroid,
            'frames_lost': 0,
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'confirmed': False,
            'uuid': str(uuid.uuid4())[:8]
        }
        
        self.person_database[person_id] = {
            'person_id': person_id,
            'uuid': self.tracks[person_id]['uuid'],
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'bbox': bbox
        }
        
        logger.info(f"New person tracked: {person_id}")
        return person_id
    
    def _update_track(self, pid: str, bbox: List[int], centroid: Tuple[int, int]):
        """Update existing track"""
        self.tracks[pid]['kalman'].update(centroid)
        self.tracks[pid]['bbox'] = bbox
        self.tracks[pid]['centroid'] = centroid
        self.tracks[pid]['frames_lost'] = 0
        self.tracks[pid]['last_seen'] = datetime.now()
        
        if self.tracks[pid]['kalman'].hits >= self.min_hits:
            self.tracks[pid]['confirmed'] = True
        
        self.person_database[pid]['last_seen'] = datetime.now().isoformat()
        self.person_database[pid]['bbox'] = bbox
    
    def _store_embedding(self, pid: str, embedding: Optional[np.ndarray]):
        """Store Re-ID embedding"""
        if embedding is None:
            return
        if pid not in self.embeddings:
            self.embeddings[pid] = deque(maxlen=10)
        self.embeddings[pid].append(embedding)
    
    def get_embedding(self, pid: str) -> Optional[np.ndarray]:
        """Get average embedding"""
        if pid not in self.embeddings or len(self.embeddings[pid]) == 0:
            return None
        return np.mean(list(self.embeddings[pid]), axis=0)
    
    def get_active_people(self) -> List[Dict]:
        """Get confirmed active people"""
        with self.lock:
            active = []
            for pid, track in self.tracks.items():
                if track['frames_lost'] == 0 and track['confirmed']:
                    vel = track['kalman'].get_velocity()
                    active.append({
                        'person_id': pid,
                        'uuid': self.person_database.get(pid, {}).get('uuid', 'N/A'),
                        'bbox': track['bbox'],
                        'first_seen': track['first_seen'].isoformat(),
                        'last_seen': track['last_seen'].isoformat(),
                        'velocity': vel.tolist() if vel is not None else [0, 0]
                    })
            return active
    
    def get_person_history(self) -> Dict:
        """Get all people ever detected"""
        with self.lock:
            return dict(self.person_database)


# ============================================================
# PERSON DETECTOR (YOLOv8 or HOG Fallback)
# ============================================================
class PersonDetector:
    """Person detection using YOLOv8 with fallback to OpenCV HOG detector"""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.hog = None
        self.method_name = "Not Initialized"
        self.initialized = False
        
        if ULTRALYTICS_AVAILABLE:
            self._load_yolov8()
        else:
            self._load_hog()
    
    def _load_yolov8(self):
        """Load YOLOv8"""
        try:
            logger.info("Loading YOLOv8n detector...")
            model_path = os.path.join(MODEL_DIR, 'yolov8n.pt')
            if not os.path.exists(model_path):
                model_path = 'yolov8n.pt'
            self.model = YOLO(model_path)
            self.method_name = "YOLOv8n"
            self.initialized = True
            logger.info("YOLOv8n detector loaded successfully")
        except Exception as e:
            logger.error(f"YOLOv8 loading error: {e}")
            self._load_hog()
    
    def _load_hog(self):
        """Load OpenCV HOG detector fallback"""
        try:
            logger.info("Loading OpenCV HOG Person Detector fallback...")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
            self.method_name = "OpenCV-HOG"
            self.initialized = True
            logger.info("OpenCV HOG loaded")
        except Exception as e:
            logger.error(f"HOG detector failed: {e}")
            self.method_name = "None"
    
    def detect(self, frame: np.ndarray) -> List[List[int]]:
        """Detect persons in frame"""
        if not self.initialized or frame is None:
            return []
        
        try:
            if self.model is not None:
                return self._detect_yolov8(frame)
            elif self.hog is not None:
                return self._detect_hog(frame)
            return []
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _detect_yolov8(self, frame: np.ndarray) -> List[List[int]]:
        """YOLOv8 detection"""
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w > 20 and h > 40:
                    boxes.append([max(0, x1), max(0, y1), w, h])
        return boxes
    
    def _detect_hog(self, frame: np.ndarray) -> List[List[int]]:
        """OpenCV HOG detection"""
        rects, weights = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        boxes = []
        for (x, y, w, h) in rects:
            if w > 20 and h > 40:
                boxes.append([int(x), int(y), int(w), int(h)])
        return boxes


# ============================================================
# VIDEO MERGER
# ============================================================
class VideoMerger:
    """Merges video segments using FFmpeg"""
    
    def __init__(self, output_dir: str = MERGED_DIR):
        self.output_dir = os.path.join(MODEL_DIR, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def merge_timeline(self, global_id: str, timeline: List[Dict], 
                       output_filename: Optional[str] = None) -> Optional[str]:
        """
        Merge timeline segments into single video
        """
        if not timeline:
            return None
            
        if output_filename is None:
            output_filename = f"{global_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            merged = self._merge_overlapping(timeline)
            concat_lines = []
            temp_files = []
            
            for seg in merged:
                segment_file = self._extract_segment(seg)
                if segment_file and os.path.exists(segment_file):
                    concat_lines.append(f"file '{segment_file}'")
                    temp_files.append(segment_file)
            
            if not concat_lines:
                return None
            
            concat_path = tempfile.mktemp(suffix='.txt')
            with open(concat_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(concat_lines))
            
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_path,
                '-c', 'copy',
                '-movflags', '+faststart',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            try:
                os.unlink(concat_path)
            except Exception:
                pass
            for f in temp_files:
                try:
                    os.unlink(f)
                except Exception:
                    pass
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Merged video created: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Merge error: {e}")
            return None
    
    def _merge_overlapping(self, timeline: List[Dict]) -> List[Dict]:
        """Merge overlapping segments"""
        if not timeline:
            return []
        
        sorted_segs = sorted(timeline, key=lambda x: x['start'])
        merged = [dict(sorted_segs[0])]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            if seg['start'] <= last['end'] + timedelta(seconds=2):
                last['end'] = max(last['end'], seg['end'])
                if 'cameras' not in last:
                    last['cameras'] = [last['camera']]
                last['cameras'].append(seg['camera'])
                last['video_file'] = seg['video_file']
            else:
                merged.append(dict(seg))
        
        return merged
    
    def _extract_segment(self, segment: Dict) -> Optional[str]:
        """Extract time segment from video"""
        src_file = segment.get('video_file', '')
        if not src_file or not os.path.exists(src_file):
            return None
        
        start = segment['start']
        end = segment['end']
        duration = (end - start).total_seconds()
        
        if duration <= 0:
            return None
        
        tmp_file = tempfile.mktemp(suffix='.mp4')
        
        cmd = [
            'ffmpeg', '-y', '-i', src_file,
            '-ss', start.strftime('%H:%M:%S.%f')[:-3],
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            tmp_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return tmp_file
        return None


# ============================================================
# CAMERA STREAM
# ============================================================
class CameraStream:
    """Manages single camera with detection, tracking, recording"""
    
    def __init__(self, camera_id: str, rtsp_url: str, 
                 cross_camera: CrossCameraIDManager,
                 reid_model: PersonReID):
        self.camera_id = camera_id
        self.rtsp_url = str(rtsp_url).strip()
        self.cross_camera = cross_camera
        self.reid = reid_model
        
        self.cap = None
        self.frame = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.stream_thread = None
        
        self.reconnect_delay = 5
        self.consecutive_failures = 0
        self.max_failures = 10
        
        # Components
        self.detector = PersonDetector(CONFIDENCE_THRESHOLD)
        self.tracker = ImprovedPersonTracker(TRACKING_MAX_DISTANCE, 
                                              TRACKING_MAX_FRAMES_LOST,
                                              MIN_HITS_TO_CONFIRM)
        
        # Detection control
        self.detection_enabled = True
        self.detection_interval = 2
        self.frame_count = 0
        self.last_detections = []
        
        # Recording
        self.recording_enabled = RECORDING_ENABLED
        self.recording_dir = os.path.join(MODEL_DIR, RECORDING_DIR, camera_id)
        os.makedirs(self.recording_dir, exist_ok=True)
        self.active_recordings: Dict[str, Any] = {}
        self.person_segments: Dict[str, List[Dict]] = {}
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def _open_capture(self):
        """Open video capture source"""
        url = self.rtsp_url.strip()
        if url.isdigit():
            # Webcam index
            return cv2.VideoCapture(int(url))
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        return cap

    def start(self) -> bool:
        """Start camera stream"""
        if self.is_running:
            return True
        
        logger.info(f"Starting camera: {self.camera_id} -> {self.rtsp_url}")
        self.cap = self._open_capture()
        
        if self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            logger.info(f"Camera {self.camera_id} connected")
        else:
            logger.warning(f"Camera {self.camera_id} not reachable immediately. Will auto-retry in background.")
            
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        return True
    
    def _stream_loop(self):
        """Main processing loop"""
        logger.info(f"Stream loop started: {self.camera_id}")
        
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(self.reconnect_delay)
                        continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self._reconnect()
                        self.consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                
                self.consecutive_failures = 0
                
                # Calculate FPS
                now = time.time()
                self.fps_counter.append(1.0 / (now - self.last_frame_time + 1e-6))
                self.last_frame_time = now
                
                # Resize
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Detection
                if self.detection_enabled and self.frame_count % self.detection_interval == 0:
                    self.last_detections = self.detector.detect(frame)
                
                # Extract embeddings
                embeddings = []
                if self.reid.model is not None:
                    for det in self.last_detections:
                        crop = self._crop_person(frame, det)
                        emb = self.reid.extract(crop)
                        embeddings.append(emb)
                
                # Tracking
                tracked = self.tracker.update(self.last_detections, embeddings)
                
                # Cross-camera matching & recording
                global_tracked = []
                for local_id, bbox in tracked:
                    local_emb = self.tracker.get_embedding(local_id)
                    global_id = self.cross_camera.match_or_create(local_emb, self.camera_id)
                    global_tracked.append((global_id, bbox, local_id))
                    
                    if self.recording_enabled:
                        self._manage_recording(global_id, frame, bbox)
                
                # Draw
                display = frame.copy()
                if self.detection_enabled:
                    self._draw_overlays(display, global_tracked)
                
                with self.lock:
                    self.frame = frame
                    self.processed_frame = display
                
                self.frame_count += 1
                time.sleep(max(0, 1.0/FPS - (time.time() - now)))
                
            except Exception as e:
                logger.error(f"Loop error ({self.camera_id}): {e}")
                time.sleep(1)
    
    def _crop_person(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop person from frame"""
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        return frame[y1:y2, x1:x2]
    
    def _manage_recording(self, global_id: str, frame: np.ndarray, bbox: List[int]):
        """Manage per-person video recording"""
        now = datetime.now()
        
        if global_id in self.active_recordings:
            rec = self.active_recordings[global_id]
            duration = (now - rec['start_time']).total_seconds()
            if duration > MAX_RECORDING_MINUTES * 60:
                self._stop_recording(global_id)
        
        if global_id not in self.active_recordings:
            filename = f"{global_id}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
            filepath = os.path.join(self.recording_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            
            self.active_recordings[global_id] = {
                'writer': writer,
                'filepath': filepath,
                'start_time': now
            }
            
            if global_id not in self.person_segments:
                self.person_segments[global_id] = []
        
        if global_id in self.active_recordings:
            self.active_recordings[global_id]['writer'].write(frame)
    
    def _stop_recording(self, global_id: str):
        """Stop and save recording"""
        if global_id not in self.active_recordings:
            return
        
        rec = self.active_recordings[global_id]
        rec['writer'].release()
        
        self.person_segments[global_id].append({
            'camera': self.camera_id,
            'start': rec['start_time'],
            'end': datetime.now(),
            'video_file': rec['filepath']
        })
        
        del self.active_recordings[global_id]
        logger.info(f"Recording saved: {global_id}")
    
    def _draw_overlays(self, frame: np.ndarray, tracked: List[Tuple[str, List[int], str]]):
        """Draw bounding boxes and info"""
        for global_id, bbox, local_id in tracked:
            x, y, w, h = bbox
            
            hash_val = int(hashlib.md5(global_id.encode()).hexdigest()[:6], 16)
            color = (hash_val & 0xFF, (hash_val >> 8) & 0xFF, (hash_val >> 16) & 0xFF)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{global_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, max(0, y - lh - 10)), (x + lw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, max(15, y - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Info overlay
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        cv2.putText(frame, f"Cam: {self.camera_id}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
        cv2.putText(frame, f"People: {len(tracked)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (FRAME_WIDTH - 170, FRAME_HEIGHT - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    
    def _get_blank_frame(self, message="Connecting...") -> np.ndarray:
        """Generate placeholder frame when camera is not ready"""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = (18, 18, 24)
        cv2.putText(frame, f"CAMERA: {self.camera_id}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 170), 2)
        cv2.putText(frame, message, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"Source: {self.rtsp_url}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now_str, (30, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1)
        return frame

    def get_frame(self, with_detection: bool = True) -> Optional[bytes]:
        """Get latest frame as JPEG bytes"""
        with self.lock:
            frame = self.processed_frame if with_detection else self.frame
            if frame is None:
                msg = "Reconnecting..." if self.consecutive_failures > 0 else "Connecting to stream..."
                frame = self._get_blank_frame(msg)
            try:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return jpeg.tobytes()
            except Exception as e:
                logger.error(f"Encode error: {e}")
                return None
    
    def get_person_segments(self, global_id: str) -> List[Dict]:
        """Get video segments for a person"""
        segments = list(self.person_segments.get(global_id, []))
        if global_id in self.active_recordings:
            rec = self.active_recordings[global_id]
            segments.append({
                'camera': self.camera_id,
                'start': rec['start_time'],
                'end': datetime.now(),
                'video_file': rec['filepath']
            })
        return segments
    
    def _reconnect(self) -> bool:
        """Reconnect to camera"""
        logger.info(f"Reconnecting camera: {self.camera_id}")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        
        time.sleep(1)
        self.cap = self._open_capture()
        if self.cap and self.cap.isOpened():
            logger.info(f"Reconnected: {self.camera_id}")
            return True
        return False
    
    def stop(self):
        """Stop camera"""
        logger.info(f"Stopping camera: {self.camera_id}")
        self.is_running = False
        
        for gid in list(self.active_recordings.keys()):
            self._stop_recording(gid)
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=3)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
    
    def toggle_detection(self) -> bool:
        """Toggle detection"""
        self.detection_enabled = not self.detection_enabled
        logger.info(f"Detection {'ON' if self.detection_enabled else 'OFF'}: {self.camera_id}")
        return self.detection_enabled


# ============================================================
# MULTI-CAMERA MANAGER
# ============================================================
class MultiCameraManager:
    """Manages all camera streams"""
    
    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}
        self.cross_camera = CrossCameraIDManager(REID_SIMILARITY_THRESHOLD)
        self.reid = PersonReID(REID_EMBEDDING_DIM, DEVICE)
        self.merger = VideoMerger(MERGED_DIR)
        self.lock = threading.Lock()
        
    def add_camera(self, camera_id: str, rtsp_url: str) -> bool:
        """Add camera"""
        with self.lock:
            if camera_id in self.cameras:
                return False
            
            cam = CameraStream(camera_id, rtsp_url, self.cross_camera, self.reid)
            self.cameras[camera_id] = cam
            cam.start()
            return True
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera"""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()
                del self.cameras[camera_id]
                return True
            return False
    
    def get_camera(self, camera_id: str) -> Optional[CameraStream]:
        """Get camera by ID"""
        return self.cameras.get(camera_id)
    
    def get_all_frames(self) -> Dict[str, bytes]:
        """Get frames from all cameras"""
        frames = {}
        for cid, cam in self.cameras.items():
            frame = cam.get_frame(with_detection=True)
            if frame:
                frames[cid] = frame
        return frames
    
    def get_person_timeline(self, global_id: str) -> List[Dict]:
        """Get complete timeline across all cameras"""
        timeline = []
        for cid, cam in self.cameras.items():
            segments = cam.get_person_segments(global_id)
            for seg in segments:
                timeline.append({
                    'camera': cid,
                    'start': seg['start'],
                    'end': seg['end'],
                    'video_file': seg['video_file']
                })
        timeline.sort(key=lambda x: x['start'])
        return timeline
    
    def merge_person_video(self, global_id: str) -> Optional[str]:
        """Merge and export person's video"""
        timeline = self.get_person_timeline(global_id)
        return self.merger.merge_timeline(global_id, timeline)
    
    def get_all_global_ids(self) -> List[str]:
        """Get all tracked global IDs"""
        return self.cross_camera.get_all_ids()
    
    def get_global_info(self, global_id: str) -> Optional[Dict]:
        """Get global ID info"""
        return self.cross_camera.get_info(global_id)
    
    def stop_all(self):
        """Stop all cameras"""
        for cam in self.cameras.values():
            cam.stop()


# ============================================================
# GLOBAL MANAGER INSTANCE
# ============================================================
manager = MultiCameraManager()


# ============================================================
# FASTAPI APPLICATION & LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    logger.info("=" * 60)
    logger.info("Multi-Camera Person Detection & Re-ID System Starting...")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Re-ID Dim: {REID_EMBEDDING_DIM}")
    logger.info(f"   Recordings: {RECORDING_DIR}")
    logger.info(f"   Merged: {MERGED_DIR}")
    logger.info("=" * 60)
    
    # Auto-start configured cameras
    for cam_config in CAMERA_CONFIGS:
        if cam_config.get('enabled') and cam_config.get('url'):
            manager.add_camera(cam_config['id'], cam_config['url'])
    
    yield
    
    logger.info("Shutting down...")
    manager.stop_all()
    logger.info("System stopped cleanly")


app = FastAPI(
    title="Multi-Camera Person Detection & Tracking",
    description="Cross-camera person tracking with Re-ID and video export",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HTML DASHBOARD TEMPLATE
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Camera Vision AI | Person Tracking & Re-ID</title>
    <style>
        :root {
            --bg: #09090d;
            --surface: #12121a;
            --surface-hover: #191924;
            --border: rgba(255, 255, 255, 0.08);
            --primary: #00d4aa;
            --primary-hover: #00b894;
            --accent: #6c5ce7;
            --text: #f0f0f5;
            --text-secondary: #9090a2;
            --success: #2ed573;
            --error: #ff4757;
            --radius: 12px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding: 24px;
        }
        .container { max-width: 1440px; margin: 0 auto; }
        
        .header {
            background: var(--surface);
            padding: 20px 28px;
            border-radius: var(--radius);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 16px;
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-title h1 {
            color: var(--primary);
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .header-actions {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(46, 213, 115, 0.1);
            color: var(--success);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(46, 213, 115, 0.2);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }
        .status-dot.offline {
            background: var(--error);
            box-shadow: 0 0 8px var(--error);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--surface);
            padding: 16px 20px;
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }
        .stat-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text);
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text);
        }

        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }
        .camera-card {
            background: var(--surface);
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .camera-header {
            padding: 12px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
        }
        .camera-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .camera-meta {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        .video-container {
            background: #000;
            aspect-ratio: 16/9;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .camera-footer {
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.01);
            border-top: 1px solid var(--border);
        }
        .camera-status-info {
            font-size: 0.82rem;
            color: var(--text-secondary);
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-decoration: none;
        }
        .btn-primary { background: var(--primary); color: #000; }
        .btn-primary:hover { background: var(--primary-hover); transform: translateY(-1px); }
        .btn-outline {
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-outline:hover {
            background: var(--surface-hover);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .btn-danger {
            background: rgba(255, 71, 87, 0.15);
            color: var(--error);
            border: 1px solid rgba(255, 71, 87, 0.3);
        }
        .btn-danger:hover {
            background: var(--error);
            color: #fff;
        }

        .people-panel {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 24px;
            border: 1px solid var(--border);
        }
        .people-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }
        .people-table th, .people-table td {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }
        .people-table th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
        }
        .people-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(0, 212, 170, 0.1);
            color: var(--primary);
            margin-right: 4px;
        }

        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: var(--surface);
            padding: 28px;
            border-radius: var(--radius);
            max-width: 480px;
            width: 90%;
            border: 1px solid var(--border);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .modal-header h3 { color: var(--primary); }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .form-group input {
            width: 100%;
            padding: 10px 14px;
            border-radius: 8px;
            background: #09090d;
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 0.9rem;
            outline: none;
        }
        .form-group input:focus {
            border-color: var(--primary);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-title">
                <h1>👁️ Multi-Camera Vision AI</h1>
                <span class="status-badge" id="systemBadge">
                    <span class="status-dot" id="systemDot"></span>
                    <span id="systemText">Online</span>
                </span>
            </div>
            <div class="header-actions">
                <button class="btn btn-outline" onclick="openAddCameraModal()">➕ Add Camera</button>
                <a href="/viewer" class="btn btn-outline">🖥️ Single Viewer</a>
                <button class="btn btn-primary" onclick="loadData()">🔄 Refresh</button>
            </div>
        </div>

        <!-- System Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Active Cameras</div>
                <div class="stat-value" id="statCameras">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total People Tracked</div>
                <div class="stat-value" id="statPeople">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Detector Engine</div>
                <div class="stat-value" id="statDetector" style="font-size: 1.25rem;">YOLOv8n</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Compute Device</div>
                <div class="stat-value" id="statDevice" style="font-size: 1.25rem;">CPU</div>
            </div>
        </div>

        <!-- Camera Streams Section -->
        <div class="section-header">
            <div class="section-title">📹 Live Camera Feeds</div>
        </div>
        <div class="camera-grid" id="cameraGrid">
            <!-- Dynamically populated -->
        </div>

        <!-- Person Tracking & Re-ID Database -->
        <div class="people-panel">
            <div class="section-header">
                <div class="section-title">👥 Cross-Camera Tracked People</div>
                <span style="font-size: 0.85rem; color: var(--text-secondary);" id="peopleCount">0 records</span>
            </div>
            <div style="overflow-x: auto;">
                <table class="people-table">
                    <thead>
                        <tr>
                            <th>Global Person ID</th>
                            <th>Cameras Observed</th>
                            <th>First Seen</th>
                            <th>Last Seen</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="peopleTableBody">
                        <tr>
                            <td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 24px;">
                                No tracked people in database yet.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Add Camera Modal -->
    <div class="modal" id="addCamModal">
        <div class="modal-content">
            <div class="modal-header">
                <h3>Add New Camera Stream</h3>
                <button class="btn btn-outline" style="padding: 4px 8px;" onclick="closeAddCameraModal()">✕</button>
            </div>
            <div class="form-group">
                <label for="newCamId">Camera ID (e.g. cam3, lobby, entrance)</label>
                <input type="text" id="newCamId" placeholder="cam3">
            </div>
            <div class="form-group">
                <label for="newCamUrl">RTSP Stream URL or Webcam Index (0, 1)</label>
                <input type="text" id="newCamUrl" placeholder="rtsp://user:pass@192.168.1.100:554/stream or 0">
            </div>
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px;">
                <button class="btn btn-outline" onclick="closeAddCameraModal()">Cancel</button>
                <button class="btn btn-primary" onclick="submitAddCamera()">Add Stream</button>
            </div>
        </div>
    </div>

    <script>
        async function loadData() {
            await Promise.all([loadCameras(), loadPeople(), loadStatus()]);
        }

        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('statCameras').textContent = data.cameras || 0;
                document.getElementById('statPeople').textContent = data.total_global_ids || 0;
                document.getElementById('statDevice').textContent = data.device ? data.device.toUpperCase() : 'CPU';
                document.getElementById('systemDot').className = 'status-dot';
                document.getElementById('systemText').textContent = 'Online';
            } catch (e) {
                document.getElementById('systemDot').className = 'status-dot offline';
                document.getElementById('systemText').textContent = 'Offline';
            }
        }

        async function loadCameras() {
            try {
                const r = await fetch('/api/cameras');
                const data = await r.json();
                const grid = document.getElementById('cameraGrid');
                
                const cameraEntries = Object.entries(data.cameras || {});
                if (cameraEntries.length === 0) {
                    grid.innerHTML = '<p style="color: var(--text-secondary); grid-column: 1/-1;">No cameras configured. Click "+ Add Camera" to start.</p>';
                    return;
                }

                grid.innerHTML = cameraEntries.map(([cid, info]) => `
                    <div class="camera-card" id="card-${cid}">
                        <div class="camera-header">
                            <div class="camera-name">
                                <span>📹</span>
                                <strong>${cid.toUpperCase()}</strong>
                            </div>
                            <div class="camera-meta">
                                ${info.method || 'YOLO'} | ${info.fps.toFixed(1)} FPS
                            </div>
                        </div>
                        <div class="video-container">
                            <img src="/video_feed/${cid}" alt="Live stream ${cid}" onerror="this.src='/video_feed/${cid}?' + Date.now()">
                        </div>
                        <div class="camera-footer">
                            <div class="camera-status-info">
                                People: <strong>${info.active_people}</strong> | Det: <strong>${info.detection ? 'ON' : 'OFF'}</strong>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-outline" style="padding: 6px 12px; font-size: 0.8rem;" onclick="toggleDetection('${cid}')">
                                    ${info.detection ? 'Disable Det' : 'Enable Det'}
                                </button>
                                <button class="btn btn-danger" style="padding: 6px 10px; font-size: 0.8rem;" onclick="removeCamera('${cid}')">
                                    🗑️
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                if (cameraEntries.length > 0) {
                    document.getElementById('statDetector').textContent = cameraEntries[0][1].method || 'YOLOv8n';
                }
            } catch (e) {
                console.error('Camera load error:', e);
            }
        }

        async function loadPeople() {
            try {
                const r = await fetch('/api/people');
                const data = await r.json();
                const tbody = document.getElementById('peopleTableBody');
                const countEl = document.getElementById('peopleCount');
                
                countEl.textContent = `${data.people ? data.people.length : 0} records`;
                
                if (!data.people || data.people.length === 0) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 24px;">
                                No tracked people in database yet.
                            </td>
                        </tr>
                    `;
                    return;
                }

                tbody.innerHTML = data.people.map(p => `
                    <tr>
                        <td><strong>👤 ${p.global_id}</strong></td>
                        <td>${p.cameras.map(c => `<span class="badge">${c}</span>`).join('')}</td>
                        <td style="color: var(--text-secondary); font-size: 0.85rem;">${new Date(p.first_seen).toLocaleTimeString()}</td>
                        <td style="color: var(--text-secondary); font-size: 0.85rem;">${new Date(p.last_seen).toLocaleTimeString()}</td>
                        <td>
                            <button class="btn btn-primary" style="padding: 6px 12px; font-size: 0.8rem;" onclick="downloadVideo('${p.global_id}')">
                                ⬇ Download Video
                            </button>
                        </td>
                    </tr>
                `).join('');
            } catch (e) {
                console.error('People load error:', e);
            }
        }

        async function toggleDetection(cid) {
            await fetch(`/api/cameras/${cid}/toggle-detection`);
            loadCameras();
        }

        async function removeCamera(cid) {
            if (confirm(`Remove camera ${cid}?`)) {
                await fetch(`/api/cameras/${cid}`, { method: 'DELETE' });
                loadData();
            }
        }

        async function downloadVideo(gid) {
            window.open(`/api/people/${gid}/download`, '_blank');
        }

        function openAddCameraModal() {
            document.getElementById('addCamModal').classList.add('active');
        }

        function closeAddCameraModal() {
            document.getElementById('addCamModal').classList.remove('active');
        }

        async function submitAddCamera() {
            const cid = document.getElementById('newCamId').value.trim();
            const url = document.getElementById('newCamUrl').value.trim();
            if (!cid || !url) {
                alert('Please provide both Camera ID and URL/Webcam Index');
                return;
            }
            try {
                const res = await fetch(`/api/cameras/${encodeURIComponent(cid)}/add?rtsp_url=${encodeURIComponent(url)}`, { method: 'POST' });
                if (res.ok) {
                    closeAddCameraModal();
                    document.getElementById('newCamId').value = '';
                    document.getElementById('newCamUrl').value = '';
                    loadData();
                } else {
                    const err = await res.json();
                    alert('Failed to add camera: ' + (err.detail || 'Unknown error'));
                }
            } catch (e) {
                alert('Error adding camera: ' + e.message);
            }
        }

        // Initial fetch and auto-polling
        loadData();
        setInterval(loadPeople, 3000);
        setInterval(loadCameras, 6000);
        setInterval(loadStatus, 5000);
    </script>
</body>
</html>
"""


# ============================================================
# API ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """Main dashboard"""
    return HTMLResponse(HTML_TEMPLATE)


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """Single camera detailed viewer interface"""
    if os.path.exists(CLIENT_INDEX):
        try:
            with open(CLIENT_INDEX, 'r', encoding='utf-8') as f:
                return HTMLResponse(f.read())
        except Exception:
            pass
    return HTMLResponse(HTML_TEMPLATE)


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """MJPEG stream for specific camera"""
    def generate():
        while True:
            cam = manager.get_camera(camera_id)
            if cam:
                frame = cam.get_frame(with_detection=True)
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/video_feed")
async def video_feed_default():
    """Default MJPEG stream for client compatibility (first camera)"""
    first_cid = next(iter(manager.cameras.keys()), "cam1")
    return await video_feed(first_cid)


@app.get("/status")
async def status():
    """Client compatibility status route"""
    active_cam = next((c for c in manager.cameras.values() if c.is_running), None)
    is_conn = (active_cam is not None and active_cam.cap is not None and active_cam.cap.isOpened())
    fps = round(float(np.mean(active_cam.fps_counter)), 1) if active_cam and active_cam.fps_counter else 0
    return {
        "connected": is_conn,
        "fps": fps,
        "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
    }


@app.get("/api/cameras")
async def get_cameras():
    """Get all camera statuses"""
    cameras = {}
    for cid, cam in manager.cameras.items():
        cameras[cid] = {
            'status': 'online' if (cam.cap and cam.cap.isOpened()) else 'offline',
            'url': cam.rtsp_url,
            'detection': cam.detection_enabled,
            'method': cam.detector.method_name,
            'active_people': len(cam.tracker.get_active_people()),
            'fps': float(np.mean(cam.fps_counter)) if cam.fps_counter else 0
        }
    return {'cameras': cameras}


@app.post("/api/cameras/{camera_id}/add")
async def add_camera(camera_id: str, rtsp_url: str):
    """Add new camera"""
    success = manager.add_camera(camera_id, rtsp_url)
    if success:
        return {'success': True, 'message': f'Camera {camera_id} added'}
    raise HTTPException(status_code=400, detail='Failed to add camera or camera already exists')


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Remove camera"""
    success = manager.remove_camera(camera_id)
    if success:
        return {'success': True, 'message': f'Camera {camera_id} removed'}
    raise HTTPException(status_code=404, detail='Camera not found')


@app.get("/api/cameras/{camera_id}/toggle-detection")
async def toggle_detection(camera_id: str):
    """Toggle detection for camera"""
    cam = manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail='Camera not found')
    enabled = cam.toggle_detection()
    return {'enabled': enabled}


@app.get("/api/people")
async def get_people():
    """Get all tracked people globally"""
    people = []
    for gid in manager.get_all_global_ids():
        info = manager.get_global_info(gid)
        if info:
            people.append(info)
    
    return {
        'count': len(people),
        'people': people
    }


@app.get("/api/people/{global_id}")
async def get_person_detail(global_id: str):
    """Get person details and timeline"""
    info = manager.get_global_info(global_id)
    if not info:
        raise HTTPException(status_code=404, detail='Person not found')
    
    timeline = manager.get_person_timeline(global_id)
    return {
        'info': info,
        'timeline': timeline,
        'segment_count': len(timeline)
    }


@app.get("/api/people/{global_id}/download")
async def download_person_video(global_id: str):
    """Download merged video for person"""
    info = manager.get_global_info(global_id)
    if not info:
        raise HTTPException(status_code=404, detail='Person not found')
    
    output_path = manager.merge_person_video(global_id)
    if output_path and os.path.exists(output_path):
        return FileResponse(
            output_path,
            media_type='video/mp4',
            filename=f"{global_id}_merged.mp4"
        )
    
    raise HTTPException(status_code=404, detail='Video not available or merge failed')


@app.get("/api/status")
async def get_status():
    """System status"""
    return {
        'system': 'online',
        'cameras': len(manager.cameras),
        'total_global_ids': len(manager.get_all_global_ids()),
        'device': DEVICE,
        'reid_available': manager.reid.model is not None
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("      MULTI-CAMERA PERSON TRACKING SYSTEM v2.0")
    print("=" * 60)
    print(f"   Device: {DEVICE}")
    print(f"   Re-ID: {REID_EMBEDDING_DIM}-dim embeddings")
    print(f"   Recordings: {RECORDING_DIR}")
    print(f"   Merged: {MERGED_DIR}")
    print()
    print("   Dashboard:     http://localhost:8000")
    print("   Single Viewer: http://localhost:8000/viewer")
    print("   API Status:    http://localhost:8000/api/status")
    print("=" * 60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)