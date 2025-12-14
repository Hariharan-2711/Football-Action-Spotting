import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from collections import deque
import os
import json
import pickle
import time
from datetime import datetime
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Data Augmentation Functions
# ============================================================================

def augment_sequence(sequence):
    """Augment a sequence with random transformations."""
    aug_sequence = sequence.copy()
    
    # Random time shift (shift by 1-3 frames)
    shift = np.random.randint(-3, 4)
    if shift != 0:
        aug_sequence = np.roll(aug_sequence, shift, axis=0)
    
    # Random noise (small gaussian noise)
    noise = np.random.normal(0, 0.01, aug_sequence.shape)
    aug_sequence = aug_sequence + noise
    
    # Random scaling (±10%)
    scale = np.random.uniform(0.9, 1.1)
    aug_sequence = aug_sequence * scale
    
    # Clip to valid range [0, 1]
    aug_sequence = np.clip(aug_sequence, 0, 1)
    
    return aug_sequence.astype(np.float32)


def balance_dataset(sequences, labels, target_count=None, augment=True):
    """Balance dataset by augmenting minority classes."""
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    label_counts = {label: np.sum(labels == label) for label in unique_labels}
    
    print(f"\n Original class distribution:")
    for label, count in label_counts.items():
        print(f"   Class {label}: {count:,} samples")
    
    if target_count is None:
        target_count = max(label_counts.values())
    
    print(f"\n Target count per class: {target_count:,}")
    
    balanced_sequences = []
    balanced_labels = []
    
    for label in unique_labels:
        class_sequences = sequences[labels == label]
        class_count = len(class_sequences)
        
        # Add original sequences
        balanced_sequences.extend(class_sequences)
        balanced_labels.extend([label] * class_count)
        
        # Augment if needed
        if class_count < target_count:
            needed = target_count - class_count
            print(f"   Class {label}: augmenting {needed:,} samples")
            
            for _ in range(needed):
                idx = np.random.randint(0, class_count)
                seq = class_sequences[idx]
                
                if augment:
                    aug_seq = augment_sequence(seq)
                else:
                    aug_seq = seq.copy()
                
                balanced_sequences.append(aug_seq)
                balanced_labels.append(label)
    
    print(f"\n Balanced dataset:")
    balanced_labels_array = np.array(balanced_labels)
    for label in unique_labels:
        count = np.sum(balanced_labels_array == label)
        print(f"   Class {label}: {count:,} samples")
    
    return balanced_sequences, balanced_labels

# Parse Ground Truth Labels
def parse_ball_action_labels(
    json_path: str, 
    labels_to_track: List[str] = None,
    label_groups: Dict[str, List[str]] = None,
    sequence_length: int = 40
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Parse Labels-ball.json with label grouping support."""
    if labels_to_track is None:
        labels_to_track = ['PASS']
    
    if label_groups is None:
        label_groups = {label: [label] for label in labels_to_track}
    
    half_seq = sequence_length // 2
    print(f"   Using sequence start = {-sequence_length} end = {sequence_length}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_labels = {}
    label_counts = {label: 0 for label in labels_to_track}
    
    for annotation in data.get('annotations', []):
        source_label = annotation.get('label', '')
        position_ms = int(annotation.get('position', 0))
        frame_idx = int(position_ms / 1000 * 25)
        
        target_label = None
        for target, sources in label_groups.items():
            if source_label in sources and target in labels_to_track:
                target_label = target
                break
        
        if target_label:
            for offset in range(-half_seq, half_seq + 1):
                frame_labels[frame_idx + offset] = target_label
            label_counts[target_label] += 1
    
    return frame_labels, label_counts

# Smart Data Extractor
class SmartSoccerNetExtractor:
    """Extract training data with label grouping."""
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        sequence_length: int = 30,
        max_players: int = 10,
        ball_proximity_threshold: float = 150.0,
        labels_to_track: List[str] = None,
        label_groups: Dict[str, List[str]] = None
    ):
        from ultralytics import YOLO
        import torch
        
        if labels_to_track is None:
            labels_to_track = ['PASS']
        
        if label_groups is None:
            label_groups = {label: [label] for label in labels_to_track}
        
        self.labels_to_track = labels_to_track
        self.label_groups = label_groups
        self.yolo = YOLO(yolo_model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            print(f"   YOLO using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   YOLO using CPU")
        
        print(f"   Tracking labels: {', '.join(labels_to_track)}")
        print(f"   Label grouping:")
        for target, sources in self.label_groups.items():
            print(f"      {target} ← {', '.join(sources)}")
        
        self.sequence_length = sequence_length
        self.max_players = max_players
        self.ball_proximity_threshold = ball_proximity_threshold
        
        self.sequences = []
        self.labels = []
        self.frame_buffer = deque(maxlen=sequence_length)

        # NEW: Track previous frame data for velocity calculation
        self.prev_frame_data = {
            'ball_position': None,
            'players': {},  # {track_id: position}
            'frame_num': 0
        }
        
        self.total_frames_processed = 0
        self.frames_with_ball_near_player = 0
        self.frames_skipped = 0
    
    def is_ball_near_player(self, ball_position, players):
        """Check if ball is close to any player."""
        if not ball_position or len(players) == 0:
            return False
        
        for player in players:
            dist = np.sqrt(
                (player['centroid'][0] - ball_position[0])**2 +
                (player['centroid'][1] - ball_position[1])**2
            )
            if dist < self.ball_proximity_threshold:
                return True
        return False

    def calculate_velocity(self, current_pos, prev_pos, time_delta=1.0):
        """
        Calculate velocity from position change.
        
        Returns: (vx, vy) normalized by frame size
        """
        if prev_pos is None or current_pos is None:
            return 0.0, 0.0
        
        vx = (current_pos[0] - prev_pos[0]) / time_delta
        vy = (current_pos[1] - prev_pos[1]) / time_delta
        
        return vx, vy

    def extract_features(self, ball_position, players, frame_size, frame_num):
        """
        Extract features with velocity.
        
        Feature vector (44D):
        - Ball: [x, y, vx, vy] = 4D
        - Players × 10: [x, y, vx, vy] × 10 = 40D
        
        Total: 44D
        """
        features = []
        width, height = frame_size
        
        # BALL FEATURES (4D)
        if ball_position:
            # Position (normalized)
            ball_x = ball_position[0] / width
            ball_y = ball_position[1] / height
            
            # Velocity (calculate from previous frame)
            if self.prev_frame_data['ball_position'] is not None:
                time_delta = frame_num - self.prev_frame_data['frame_num']
                if time_delta > 0:
                    prev_ball = self.prev_frame_data['ball_position']
                    vx = (ball_position[0] - prev_ball[0]) / width / time_delta
                    vy = (ball_position[1] - prev_ball[1]) / height / time_delta
                else:
                    vx, vy = 0.0, 0.0
            else:
                vx, vy = 0.0, 0.0
            
            features.extend([ball_x, ball_y, vx, vy])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # PLAYER FEATURES (40D = 10 players × 4D)
        
        # Sort players by distance to ball
        if ball_position and players:
            players_sorted = sorted(
                players,
                key=lambda p: np.sqrt(
                    (p['centroid'][0] - ball_position[0])**2 +
                    (p['centroid'][1] - ball_position[1])**2
                )
            )
        else:
            players_sorted = players
        
        for i in range(self.max_players):
            if i < len(players_sorted):
                player = players_sorted[i]
                pos = player['centroid']
                track_id = player['track_id']
                
                # Position (normalized)
                px = pos[0] / width
                py = pos[1] / height
                
                # Velocity (calculate from previous frame)
                if track_id in self.prev_frame_data['players']:
                    time_delta = frame_num - self.prev_frame_data['frame_num']
                    if time_delta > 0:
                        prev_pos = self.prev_frame_data['players'][track_id]
                        vx = (pos[0] - prev_pos[0]) / width / time_delta
                        vy = (pos[1] - prev_pos[1]) / height / time_delta
                    else:
                        vx, vy = 0.0, 0.0
                else:
                    # First time seeing this player
                    vx, vy = 0.0, 0.0
                
                features.extend([px, py, vx, vy])
            else:
                # Padding for missing players
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # UPDATE PREVIOUS FRAME DATA
        self.prev_frame_data['ball_position'] = ball_position
        self.prev_frame_data['players'] = {
            p['track_id']: p['centroid'] for p in players_sorted
        }
        self.prev_frame_data['frame_num'] = frame_num
        
        return np.array(features, dtype=np.float32)
    
    def process_video(self, video_path, labels_path, max_frames=None):
        """Process video with smart frame selection."""
        
        print(f"\n📹 Processing: {os.path.basename(video_path)}")
        print(f"   Labels: {os.path.basename(labels_path)}")
        
        if not os.path.exists(labels_path):
            print(f"   Labels not found, skipping")
            return
        
        frame_labels, label_counts = parse_ball_action_labels(
            labels_path, 
            self.labels_to_track,
            self.label_groups,
            self.sequence_length
        )
        
        print(f"   Ground truth labels (after grouping):")
        for label in self.labels_to_track:
            count = label_counts.get(label, 0)
            frame_count = sum(1 for v in frame_labels.values() if v == label)
            print(f"      {label}: {count} events ({frame_count} frames)")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"   Cannot open video")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   {width}x{height} @ {fps}fps, {total_frames} frames")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"   Limited to: {max_frames} frames")
        
        frame_num = 0
        actions_found = 0
        background_found = 0
        video_frames_processed = 0
        video_frames_skipped = 0
        label_counters = {label: 0 for label in self.labels_to_track}
        
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                if max_frames and frame_num > max_frames:
                    break
                
                self.total_frames_processed += 1
                video_frames_processed += 1
                
                results_players = self.yolo.track(
                    frame, persist=True, tracker='bytetrack.yaml',
                    classes=[0], conf=0.15, iou=0.45, max_det=22,
                    verbose=False, device=self.device
                )[0]
                
                results_ball = self.yolo(
                    frame, classes=[32], conf=0.02, verbose=False, device=self.device
                )[0]
                
                players = []
                if results_players.boxes is not None:
                    for i, box in enumerate(results_players.boxes):
                        bbox = box.xyxy[0].cpu().numpy()
                        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        track_id = int(box.id[0]) if box.id is not None else i
                        
                        players.append({
                            'track_id': track_id,
                            'centroid': centroid,
                            'bbox': bbox
                        })
                
                ball_position = None
                if results_ball.boxes is not None and len(results_ball.boxes) > 0:
                    best_box = max(results_ball.boxes, key=lambda b: float(b.conf[0]))
                    bbox = best_box.xyxy[0].cpu().numpy()
                    ball_position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
               
                # Frame filtering
                #if not self.is_ball_near_player(ball_position, players):
                #    self.frames_skipped += 1
                #    video_frames_skipped += 1
                #    continue
                
                self.frames_with_ball_near_player += 1
                
                if ball_position and len(players) > 0:
                    features = self.extract_features(ball_position, players, (width, height), frame_num)
                    self.frame_buffer.append((features, frame_num))
                    
                    if len(self.frame_buffer) == self.sequence_length:
                        center_frame = self.frame_buffer[self.sequence_length // 2][1]
                        frame_label = frame_labels.get(center_frame)
                        
                        sequence = np.array([f for f, _ in self.frame_buffer], dtype=np.float32)
                        
                        if frame_label in self.labels_to_track:
                            self.sequences.append(sequence)
                            label_idx = self.labels_to_track.index(frame_label)
                            self.labels.append(label_idx)
                            label_counters[frame_label] += 1
                            actions_found += 1
                        elif np.random.random() < 0.5:
                            self.sequences.append(sequence)
                            background_idx = self.labels_to_track.index('BACKGROUND')
                            self.labels.append(background_idx)
                            background_found += 1
                
                if frame_num % 2000 == 0:
                    progress = 100 * frame_num / total_frames
                    elapsed = time.time() - start_time
                    fps_rate = frame_num / elapsed if elapsed > 0 else 0
                    relevance = 100 * (video_frames_processed - video_frames_skipped) / video_frames_processed if video_frames_processed > 0 else 0
                    
                    label_str = ", ".join([f"{label}: {label_counters[label]}" for label in self.labels_to_track])
                    print(f"   Progress: {progress:.1f}% - {label_str}, Background: {background_found}, Relevant: {relevance:.1f}%, Speed: {fps_rate:.1f} fps")
        
        except KeyboardInterrupt:
            print("\n   Interrupted")
        finally:
            cap.release()
        
        elapsed = time.time() - start_time
        relevance = 100 * (video_frames_processed - video_frames_skipped) / video_frames_processed if video_frames_processed > 0 else 0
        
        total_sequences = actions_found + background_found
        print(f"   Extracted {total_sequences} sequences in {elapsed:.1f}s")
        print(f"   Label breakdown:")
        for label in self.labels_to_track:
            print(f"      {label}: {label_counters[label]}")
        print(f"      BACKGROUND: {background_found}")
        print(f"   Relevant: {relevance:.1f}% ({video_frames_processed - video_frames_skipped}/{video_frames_processed})")


# LSTM Model
class PassDetectionLSTM(nn.Module):
    def __init__(self, input_size=44, hidden_size=128, num_layers=2, num_classes=2):
        super(PassDetectionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class RealVideoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )

# Metrics Computation
def compute_metrics(model, data_loader, device, num_classes, class_names):
    """Compute comprehensive evaluation metrics."""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Accuracy
    accuracy = (all_predictions == all_labels).mean()
    
    # F1 Score
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Precision and Recall
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # mAP
    ap_scores = []
    for i in range(num_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_scores = all_probabilities[:, i]
        
        if y_true_binary.sum() > 0:
            ap = average_precision_score(y_true_binary, y_scores)
            ap_scores.append(ap)
        else:
            ap_scores.append(0.0)
    
    map_score = np.mean(ap_scores)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro,
        'precision_per_class': precision_per_class,
        'recall_macro': recall_macro,
        'recall_per_class': recall_per_class,
        'confusion_matrix': cm,
        'mAP': map_score,
        'AP_per_class': ap_scores
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix saved: {save_path}")


def plot_training_curves(train_losses, train_accs, test_losses, test_accs, save_path='training_curves.png'):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Training curves saved: {save_path}")


def save_metrics_report(metrics, class_names, save_path='metrics_report.txt'):
    """Save detailed metrics report."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 20 + "EVALUATION METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"F1 Score (Macro):   {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (Macro):  {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):     {metrics['recall_macro']:.4f}\n")
        f.write(f"mAP (Mean AP):      {metrics['mAP']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<15} {'F1':<10} {'Precision':<12} {'Recall':<10} {'AP':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, class_name in enumerate(class_names):
            if i < len(metrics['f1_per_class']):
                f.write(f"{class_name:<15} "
                       f"{metrics['f1_per_class'][i]:<10.4f} "
                       f"{metrics['precision_per_class'][i]:<12.4f} "
                       f"{metrics['recall_per_class'][i]:<10.4f} "
                       f"{metrics['AP_per_class'][i]:<10.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'':>15}")
        for name in class_names:
            f.write(f"{name:>12}")
        f.write("\n")
        
        cm = metrics['confusion_matrix']
        for i, name in enumerate(class_names):
            if i < len(cm):
                f.write(f"{name:>15}")
                for j in range(len(class_names)):
                    if j < len(cm[i]):
                        f.write(f"{cm[i][j]:>12d}")
                f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   Metrics report saved: {save_path}")


# Training with Train/Test Split
def train_smart_fixed_split(
    data_root: str,
    output_model: str = "fa_detector_no_leakage.pth",
    num_epochs: int = 50,
    batch_size: int = 32,
    max_frames_per_video: int = None,
    ball_proximity: float = 150.0,
    save_every: int = 10,
    use_cached_data: bool = True,
    train_cache_file: str = "train_extracted_data.pkl",
    test_cache_file: str = "test_extracted_data.pkl",
    labels_to_track: List[str] = None,
    label_groups: Dict[str, List[str]] = None,
    balance_classes: bool = True,
    augment_data: bool = True,
    output_dir: str = "training_results",
    sequence_length: int = 40
):
    
    if labels_to_track is None:
        labels_to_track = ['PASS', 'DRIVE', 'BACKGROUND']
    
    if label_groups is None:
        label_groups = {label: [label] for label in labels_to_track}
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime.now()
    
    print("=" * 70)
    print(" " * 10 + "TRAINING WITH TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output model: {output_model}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Labels to track: {', '.join(labels_to_track)}")
    print(f"Label grouping:")
    for target, sources in label_groups.items():
        print(f"  {target} ← {', '.join(sources)}")
    print("=" * 70)
    
    # STEP 1: EXTRACT TRAINING DATA (from train/ folder)
    
    train_sequences = None
    train_labels = None
    
    if use_cached_data and os.path.exists(train_cache_file):
        print(f"\nFound cached training data: {train_cache_file}")
        try:
            with open(train_cache_file, 'rb') as f:
                cached = pickle.load(f)
            train_sequences = cached['sequences']
            train_labels = cached['labels']
            print(f"Loaded training data from cache!")
            print(f"   Training sequences: {len(train_sequences):,}")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            train_sequences = None
            train_labels = None
    
    if train_sequences is None or train_labels is None:
        print(f"\n{'='*70}")
        print("STEP 1: EXTRACT TRAINING DATA (train/ folder)")
        print("=" * 70)
        
        train_dir = os.path.join(data_root, 'train')
        train_videos = []
        
        for root, dirs, files in os.walk(train_dir):
            if "224p.mp4" in files and "Labels-ball.json" in files:
                video_path = os.path.join(root, "224p.mp4")
                labels_path = os.path.join(root, "Labels-ball.json")
                video_name = os.path.basename(os.path.dirname(root))
                train_videos.append((video_path, labels_path, video_name))
        
        print(f"\n📂 Found {len(train_videos)} training videos:")
        for vp, lp, vn in train_videos:
            print(f"   - {vn}")
        
        if len(train_videos) == 0:
            print(f"\nNo training videos found in {train_dir}")
            return None
        
        extractor = SmartSoccerNetExtractor(
            ball_proximity_threshold=ball_proximity,
            labels_to_track=labels_to_track,
            label_groups=label_groups,
            sequence_length=sequence_length
        )
        
        extraction_start = time.time()
        for i, (video_path, labels_path, video_name) in enumerate(train_videos, 1):
            print(f"\n[Training Video {i}/{len(train_videos)}]")
            extractor.process_video(video_path, labels_path, max_frames=max_frames_per_video)
        
        extraction_time = time.time() - extraction_start
        
        if len(extractor.sequences) == 0:
            print("\nNo training data extracted!")
            return None
        
        train_sequences = extractor.sequences
        train_labels = extractor.labels
        
        total_relevance = 100 * extractor.frames_with_ball_near_player / extractor.total_frames_processed if extractor.total_frames_processed > 0 else 0
        
        print(f"\n{'='*70}")
        print("TRAINING EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"Time taken: {extraction_time:.1f}s ({extraction_time/60:.1f} minutes)")
        print(f"Total frames: {extractor.total_frames_processed:,}")
        print(f"Relevant frames: {extractor.frames_with_ball_near_player:,} ({total_relevance:.1f}%)")
        print(f"Skipped frames: {extractor.frames_skipped:,} ({100-total_relevance:.1f}%)")
        print(f"Training sequences: {len(train_sequences):,}")
        
        labels_array = np.array(train_labels)
        for i, label in enumerate(labels_to_track):
            count = np.sum(labels_array == i)
            percentage = 100 * count / len(train_labels) if len(train_labels) > 0 else 0
            print(f"  ├─ {label}: {count:,} ({percentage:.1f}%)")
        
        with open(train_cache_file, 'wb') as f:
            pickle.dump({'sequences': train_sequences, 'labels': train_labels}, f)
        print(f"Training data cached: {train_cache_file}")
    
    # STEP 2: EXTRACT TEST DATA (from test/ folder)
    
    test_sequences = None
    test_labels = None
    
    if use_cached_data and os.path.exists(test_cache_file):
        print(f"\nFound cached test data: {test_cache_file}")
        try:
            with open(test_cache_file, 'rb') as f:
                cached = pickle.load(f)
            test_sequences = cached['sequences']
            test_labels = cached['labels']
            print(f" Loaded test data from cache!")
            print(f"   Test sequences: {len(test_sequences):,}")
        except Exception as e:
            print(f"  Failed to load cache: {e}")
            test_sequences = None
            test_labels = None
    
    if test_sequences is None or test_labels is None:
        print(f"\n{'='*70}")
        print("STEP 2: EXTRACT TEST DATA (test/ folder)")
        print("=" * 70)
        
        test_dir = os.path.join(data_root, 'test')
        test_videos = []
        
        for root, dirs, files in os.walk(test_dir):
            if "224p.mp4" in files and "Labels-ball.json" in files:
                video_path = os.path.join(root, "224p.mp4")
                labels_path = os.path.join(root, "Labels-ball.json")
                video_name = os.path.basename(os.path.dirname(root))
                test_videos.append((video_path, labels_path, video_name))
        
        print(f"\nFound {len(test_videos)} test videos:")
        for vp, lp, vn in test_videos:
            print(f"   - {vn}")
        
        if len(test_videos) == 0:
            print(f"\n No test videos found in {test_dir}")
            return None
        
        # Create new extractor for test data
        test_extractor = SmartSoccerNetExtractor(
            ball_proximity_threshold=ball_proximity,
            labels_to_track=labels_to_track,
            label_groups=label_groups,
            sequence_length=sequence_length
        )
        
        extraction_start = time.time()
        for i, (video_path, labels_path, video_name) in enumerate(test_videos, 1):
            print(f"\n[Test Video {i}/{len(test_videos)}]")
            test_extractor.process_video(video_path, labels_path, max_frames=max_frames_per_video)
        
        extraction_time = time.time() - extraction_start
        
        if len(test_extractor.sequences) == 0:
            print("\n No test data extracted!")
            return None
        
        test_sequences = test_extractor.sequences
        test_labels = test_extractor.labels
        
        total_relevance = 100 * test_extractor.frames_with_ball_near_player / test_extractor.total_frames_processed if test_extractor.total_frames_processed > 0 else 0
        
        print(f"\n{'='*70}")
        print("TEST EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"Time taken: {extraction_time:.1f}s ({extraction_time/60:.1f} minutes)")
        print(f"Total frames: {test_extractor.total_frames_processed:,}")
        print(f"Relevant frames: {test_extractor.frames_with_ball_near_player:,} ({total_relevance:.1f}%)")
        print(f"Skipped frames: {test_extractor.frames_skipped:,} ({100-total_relevance:.1f}%)")
        print(f"Test sequences: {len(test_sequences):,}")
        
        labels_array = np.array(test_labels)
        for i, label in enumerate(labels_to_track):
            count = np.sum(labels_array == i)
            percentage = 100 * count / len(test_labels) if len(test_labels) > 0 else 0
            print(f"  ├─ {label}: {count:,} ({percentage:.1f}%)")
        
        with open(test_cache_file, 'wb') as f:
            pickle.dump({'sequences': test_sequences, 'labels': test_labels}, f)
        print(f" Test data cached: {test_cache_file}")
    
    print("=" * 70)
    
    # STEP 3: CLASS BALANCING (training data only)
    
    if balance_classes:
        print(f"\n{'='*70}")
        print("STEP 3: CLASS BALANCING (TRAINING DATA ONLY)")
        print("=" * 70)
        train_sequences, train_labels = balance_dataset(
            train_sequences, 
            train_labels, 
            augment=augment_data
        )
    
    # STEP 4: CREATE DATASETS
    
    print(f"\n{'='*70}")
    print("STEP 4: CREATE PYTORCH DATASETS")
    print("=" * 70)
    
    train_dataset = RealVideoDataset(train_sequences, train_labels)
    test_dataset = RealVideoDataset(test_sequences, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train set: {len(train_dataset):,} sequences")
    print(f"Test set: {len(test_dataset):,} sequences")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # STEP 5: MODEL TRAINING
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    num_classes = len(labels_to_track)
    class_names = labels_to_track
    
    model = PassDetectionLSTM(
        input_size=44, 
        hidden_size=128, 
        num_layers=2,
        num_classes=num_classes
    )
    model = model.to(device)
    
    print(f"\nModel has {num_classes} output classes (FIXED!)")
    print(f"Classes: {', '.join(class_names)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n{'='*70}")
    print("TRAINING PROGRESS")
    print("=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12} {'Test F1':<10}")
    print("-" * 70)
    
    best_test_f1 = 0.0
    best_test_acc = 0.0
    training_start = time.time()
    
    train_losses_history = []
    train_accs_history = []
    test_losses_history = []
    test_accs_history = []
    test_f1_history = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences_batch, labels_batch in train_loader:
            sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels_batch).sum().item()
            train_total += labels_batch.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for sequences_batch, labels_batch in test_loader:
                sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
                outputs = model(sequences_batch)
                loss = criterion(outputs, labels_batch)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels_batch).sum().item()
                test_total += labels_batch.size(0)
        
        test_loss /= len(test_loader) if len(test_loader) > 0 else 1
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        test_metrics = compute_metrics(model, test_loader, device, num_classes, class_names)
        test_f1 = test_metrics['f1_macro']
        
        train_losses_history.append(train_loss)
        train_accs_history.append(train_acc)
        test_losses_history.append(test_loss)
        test_accs_history.append(test_acc)
        test_f1_history.append(test_f1)
        
        epoch_time = time.time() - epoch_start
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f} {test_f1:<10.4f}")
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'labels': labels_to_track,
                'label_groups': label_groups,
                'num_classes': num_classes,
                'input_size': 44,
                'hidden_size': 128,
                'num_layers': 2,
                'sequence_length': sequence_length,
                'epoch': epoch + 1,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'train_acc': train_acc
            }
            torch.save(save_dict, output_model)
            print(f"         New best model! (F1: {test_f1:.4f}, Acc: {test_acc:.4f})")
        
        if (epoch + 1) % save_every == 0:
            checkpoint_name = f"{output_model.replace('.pth', '')}_epoch{epoch+1}.pth"
            save_dict = {
                'model_state_dict': model.state_dict(),
                'labels': labels_to_track,
                'label_groups': label_groups,
                'num_classes': num_classes,
                'epoch': epoch + 1,
                'test_acc': test_acc,
                'test_f1': test_f1
            }
            torch.save(save_dict, checkpoint_name)
            print(f"         Checkpoint saved: {checkpoint_name}")
    
    training_time = time.time() - training_start
    
    # FINAL EVALUATION
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    train_metrics = compute_metrics(model, train_loader, device, num_classes, class_names)
    test_metrics = compute_metrics(model, test_loader, device, num_classes, class_names)
    
    print("\nTRAIN METRICS:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {train_metrics['f1_macro']:.4f}")
    print(f"  mAP:      {train_metrics['mAP']:.4f}")
    
    print("\nTEST METRICS (UNSEEN VIDEOS!):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_macro']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  mAP:       {test_metrics['mAP']:.4f}")
    
    print("\nPER-CLASS METRICS (Test Set):")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    F1:        {test_metrics['f1_per_class'][i]:.4f}")
        print(f"    Precision: {test_metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall:    {test_metrics['recall_per_class'][i]:.4f}")
        print(f"    AP:        {test_metrics['AP_per_class'][i]:.4f}")
    
    # Save visualizations
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, cm_path)
    
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(train_losses_history, train_accs_history, 
                        test_losses_history, test_accs_history, curves_path)
    
    report_path = os.path.join(output_dir, 'test_metrics_report.txt')
    save_metrics_report(test_metrics, class_names, report_path)
    
    # Save metrics as JSON
    metrics_dict = {
        'train': {
            'accuracy': float(train_metrics['accuracy']),
            'f1_macro': float(train_metrics['f1_macro']),
            'mAP': float(train_metrics['mAP'])
        },
        'test': {
            'accuracy': float(test_metrics['accuracy']),
            'f1_macro': float(test_metrics['f1_macro']),
            'precision_macro': float(test_metrics['precision_macro']),
            'recall_macro': float(test_metrics['recall_macro']),
            'mAP': float(test_metrics['mAP'])
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics_dict['per_class'][class_name] = {
            'f1': float(test_metrics['f1_per_class'][i]),
            'precision': float(test_metrics['precision_per_class'][i]),
            'recall': float(test_metrics['recall_per_class'][i]),
            'AP': float(test_metrics['AP_per_class'][i])
        }
    
    json_path = os.path.join(output_dir, 'test_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"   Metrics JSON saved: {json_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best test F1: {best_test_f1:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print(f"Model saved: {output_model}")
    print(f"Results saved in: {output_dir}")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM with proper train/test split')
    parser.add_argument('--data-root', type=str, default='./SoccerNet/SN-BAS-2025')
    parser.add_argument('--output', type=str, default='fa_detector_no_leakage.pth')
    parser.add_argument('--output-dir', type=str, default='training_results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--ball-proximity', type=float, default=150.0)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--train-cache', type=str, default='train_extracted_data.pkl')
    parser.add_argument('--test-cache', type=str, default='test_extracted_data.pkl')
    parser.add_argument('--force-extract', action='store_true')
    parser.add_argument('--labels', type=str, nargs='+', default=['PASS', 'DRIVE', 'BACKGROUND'])
    parser.add_argument('--label-groups', type=str, default=None)
    parser.add_argument('--no-balance', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--sequence-length', type=int, default=40)
    
    args = parser.parse_args()
    
    label_groups = None
    if args.label_groups:
        try:
            label_groups = json.loads(args.label_groups)
        except json.JSONDecodeError as e:
            print(f"Error parsing --label-groups: {e}")
            exit(1)
    else:
        # Default label grouping
        label_groups = {
            'PASS': ['PASS', 'HIGH PASS', 'HEADER', 'CROSS'],
            'DRIVE': ['DRIVE'],
            'BACKGROUND': ['BALL PLAYER BLOCK', 'FREE KICK', 'GOAL', 'OUT', 
                          'PLAYER SUCCESSFUL TACKLE', 'SHOT', 'THROW IN']
        }
    
    train_smart_fixed_split(
        data_root=args.data_root,
        output_model=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_frames_per_video=args.max_frames,
        ball_proximity=args.ball_proximity,
        save_every=args.save_every,
        use_cached_data=args.use_cache and not args.force_extract,
        train_cache_file=args.train_cache,
        test_cache_file=args.test_cache,
        labels_to_track=args.labels,
        label_groups=label_groups,
        balance_classes=not args.no_balance,
        augment_data=not args.no_augment,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length
    )
