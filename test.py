import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
import argparse
import time
import json
import os
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


class PassDetectionLSTM(nn.Module):
    """LSTM model for pass detection."""
    
    def __init__(self, input_size=44, hidden_size=128, num_layers=2, num_classes=2, use_batchnorm=False):
        super(PassDetectionLSTM, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.input_bn = nn.BatchNorm1d(input_size)
        
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
        # Apply BatchNorm if enabled
        if self.use_batchnorm:
            # x: (batch, sequence, features)
            x = x.transpose(1, 2)  # (batch, features, sequence)
            x = self.input_bn(x)
            x = x.transpose(1, 2)  # (batch, sequence, features)
        
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def parse_ground_truth_labels(
    labels_path: str,
    labels_to_track: list,
    label_groups: dict = None,
    fps: int = 25,
    sequence_length: int = 40
):
    """
    Parse ground truth labels with label grouping.
    Uses sequence_length to determine labeling window.
    """
    if not os.path.exists(labels_path):
        return None, None
    
    if label_groups is None:
        label_groups = {label: [label] for label in labels_to_track}
    
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    frame_labels = {}
    label_counts = {label: 0 for label in labels_to_track}
    source_counts = {}
    
    half_seq = sequence_length // 2
    
    for annotation in data.get('annotations', []):
        source_label = annotation.get('label', '')
        position_ms = int(annotation.get('position', 0))
        frame_idx = int(position_ms / 1000 * fps)
        
        source_counts[source_label] = source_counts.get(source_label, 0) + 1
        
        target_label = None
        for target, sources in label_groups.items():
            if source_label in sources and target in labels_to_track:
                target_label = target
                break
        
        if target_label:
            for offset in range(-half_seq, half_seq + 1):
                frame_labels[frame_idx + offset] = target_label
            label_counts[target_label] += 1
    
    print(f"   Ground truth events (after grouping):")
    for target in labels_to_track:
        sources = label_groups.get(target, [target])
        source_breakdown = []
        for src in sources:
            if src in source_counts and source_counts[src] > 0:
                source_breakdown.append(f"{src}({source_counts[src]})")
        
        if label_counts[target] > 0:
            breakdown_str = ', '.join(source_breakdown) if source_breakdown else 'none'
            print(f"      {target}: {label_counts[target]} events [{breakdown_str}]")
    
    return frame_labels, label_counts


def compute_detection_metrics(detected_events, ground_truth_labels, total_frames, labels, tolerance=15):
    """Compute frame-level and event-level metrics."""
    if ground_truth_labels is None:
        return None
    
    predictions = {}
    ground_truth = {}
    
    for frame in range(1, total_frames + 1):
        predictions[frame] = 'BACKGROUND'
        ground_truth[frame] = ground_truth_labels.get(frame, 'BACKGROUND')
    
    for event in detected_events:
        frame = event['frame']
        label = event['label']
        for offset in range(-tolerance, tolerance + 1):
            if 1 <= frame + offset <= total_frames:
                predictions[frame + offset] = label
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    if 'BACKGROUND' not in label_to_idx:
        label_to_idx['BACKGROUND'] = len(labels)
    
    y_true = [label_to_idx.get(ground_truth[f], label_to_idx['BACKGROUND']) 
              for f in range(1, total_frames + 1)]
    y_pred = [label_to_idx.get(predictions[f], label_to_idx['BACKGROUND']) 
              for f in range(1, total_frames + 1)]
    
    unique_labels = sorted(set(y_true + y_pred))
    all_label_names = labels + ['BACKGROUND'] if 'BACKGROUND' not in labels else labels
    
    metrics = {
        'accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_per_class': {},
        'precision_per_class': {},
        'recall_per_class': {},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=unique_labels)
    }
    
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=unique_labels)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=unique_labels)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=unique_labels)
    
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    for i, label_idx in enumerate(unique_labels):
        label_name = idx_to_label[label_idx]
        if i < len(f1_per_class):
            metrics['f1_per_class'][label_name] = f1_per_class[i]
            metrics['precision_per_class'][label_name] = precision_per_class[i]
            metrics['recall_per_class'][label_name] = recall_per_class[i]
    
    for label in all_label_names:
        if label not in metrics['f1_per_class']:
            metrics['f1_per_class'][label] = 0.0
            metrics['precision_per_class'][label] = 0.0
            metrics['recall_per_class'][label] = 0.0
    
    gt_events = {}
    for frame, label in ground_truth_labels.items():
        if label != 'BACKGROUND':
            found = False
            for event_frame in gt_events:
                if abs(frame - event_frame) <= tolerance and gt_events[event_frame] == label:
                    found = True
                    break
            if not found:
                gt_events[frame] = label
    
    true_positives = 0
    false_positives = 0
    matched_gt = set()
    
    for detected in detected_events:
        matched = False
        for gt_frame, gt_label in gt_events.items():
            if (abs(detected['frame'] - gt_frame) <= tolerance and 
                detected['label'] == gt_label and 
                gt_frame not in matched_gt):
                true_positives += 1
                matched_gt.add(gt_frame)
                matched = True
                break
        if not matched:
            false_positives += 1
    
    false_negatives = len(gt_events) - len(matched_gt)
    
    event_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    event_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0
    
    metrics['event_level'] = {
        'precision': event_precision,
        'recall': event_recall,
        'f1': event_f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'ground_truth_events': len(gt_events),
        'detected_events': len(detected_events)
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix saved: {save_path}")


def save_test_report(metrics, class_names, save_path):
    """Save detailed test report."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 25 + "TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("FRAME-LEVEL METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"F1 Score (Macro):  {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):    {metrics['recall_macro']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<15} {'F1':<10} {'Precision':<12} {'Recall':<10}\n")
        f.write("-" * 70 + "\n")
        for label in class_names:
            f.write(f"{label:<15} "
                   f"{metrics['f1_per_class'][label]:<10.4f} "
                   f"{metrics['precision_per_class'][label]:<12.4f} "
                   f"{metrics['recall_per_class'][label]:<10.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("EVENT-LEVEL METRICS:\n")
        f.write("-" * 70 + "\n")
        event = metrics['event_level']
        f.write(f"Precision: {event['precision']:.4f}\n")
        f.write(f"Recall:    {event['recall']:.4f}\n")
        f.write(f"F1 Score:  {event['f1']:.4f}\n\n")
        f.write(f"True Positives:  {event['true_positives']}\n")
        f.write(f"False Positives: {event['false_positives']}\n")
        f.write(f"False Negatives: {event['false_negatives']}\n")
        f.write(f"Ground Truth:    {event['ground_truth_events']}\n")
        f.write(f"Detected:        {event['detected_events']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   Test report saved: {save_path}")


def test_pass_detection(
    video_path: str,
    model_path: str,
    output_path: str = None,
    confidence_threshold: float = 0.6,
    display: bool = True,
    yolo_model: str = 'yolov8n.pt',
    labels: list = None,
    num_classes: int = None,
    override_label_groups: dict = None,
    ground_truth_path: str = None,
    output_dir: str = "test_results"
):
    """
    Test LSTM with detection + comprehensive metrics.
    UPDATED: Supports velocity features (44D)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(" " * 15 + "PASS DETECTION TEST WITH METRICS")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract metadata from checkpoint
    sequence_length = 40
    input_size = 44  # Default
    use_batchnorm = False
    label_groups = None
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if 'sequence_length' in checkpoint:
            sequence_length = checkpoint['sequence_length']
            print(f" Sequence length from model: {sequence_length}")
        else:
            print(f" No sequence_length in checkpoint, using default: {sequence_length}")
        
        # ✅ NEW: Get input_size from checkpoint
        if 'input_size' in checkpoint:
            input_size = checkpoint['input_size']
            print(f" Input size: {input_size}D {'(with velocity)' if input_size == 44 else ''}")
        else:
            print(f"  No input_size in checkpoint, using default: {input_size}D")
        
        # ✅ NEW: Get batchnorm setting
        if 'use_batchnorm' in checkpoint:
            use_batchnorm = checkpoint['use_batchnorm']
            print(f" BatchNorm: {'enabled' if use_batchnorm else 'disabled'}")
        
        if labels is None:
            labels = checkpoint.get('labels', ['PASS'])
            print(f" Auto-detected labels: {', '.join(labels)}")
        
        if 'label_groups' in checkpoint:
            label_groups = checkpoint['label_groups']
            print(f" Label grouping from model:")
            for target, sources in label_groups.items():
                print(f"      {target} ← {', '.join(sources)}")
        else:
            label_groups = {label: [label] for label in labels}
            print(f"  No label grouping in model (1:1 mapping)")
        
        if override_label_groups is not None:
            label_groups = override_label_groups
            print(f"  OVERRIDING label grouping:")
            for target, sources in label_groups.items():
                print(f"      {target} ← {', '.join(sources)}")
        
        if num_classes is None:
            num_classes = checkpoint.get('num_classes', len(labels))
        
        model_state = checkpoint['model_state_dict']
        
        if 'test_acc' in checkpoint:
            print(f" Model test accuracy: {checkpoint['test_acc']:.4f}")
        if 'test_f1' in checkpoint:
            print(f" Model test F1: {checkpoint['test_f1']:.4f}")
        elif 'val_acc' in checkpoint:
            print(f" Model val accuracy: {checkpoint['val_acc']:.4f}")
        if 'val_f1' in checkpoint:
            print(f" Model val F1: {checkpoint['val_f1']:.4f}")
    else:
        model_state = checkpoint
        print("  Old model format")
        print(f"  Using defaults: sequence_length={sequence_length}, input_size={input_size}")
        
        if labels is None:
            labels = ['PASS']
        
        if override_label_groups is not None:
            label_groups = override_label_groups
        else:
            label_groups = {label: [label] for label in labels}
        
        if num_classes is None:
            num_classes = len(labels)
    
    label_names = labels if 'BACKGROUND' in labels else labels + ['BACKGROUND']
    
    print(f"Target labels: {', '.join(labels)}")
    print(f"Num classes: {num_classes}")
    
    # Create model with correct input_size
    model = PassDetectionLSTM(
        input_size=input_size,  # Use from checkpoint (32 or 44)
        hidden_size=128, 
        num_layers=2,
        num_classes=num_classes,
        use_batchnorm=use_batchnorm
    )
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded: {model_path}")
    print(f"   Confidence threshold: {confidence_threshold}")
    print(f"   Sequence length: {sequence_length} frames")
    print(f"   Feature dimensions: {input_size}D")
    
    yolo = YOLO(yolo_model)
    print(f" YOLO loaded: {yolo_model}")
    
    ground_truth_labels = None
    ground_truth_counts = None
    
    if ground_truth_path and os.path.exists(ground_truth_path):
        print(f" Ground truth: {ground_truth_path}")
        print(f"   Applying label grouping:")
        for target, sources in label_groups.items():
            print(f"      {target} ← {', '.join(sources)}")
        
        ground_truth_labels, ground_truth_counts = parse_ground_truth_labels(
            ground_truth_path,
            labels,
            label_groups,
            sequence_length=sequence_length
        )
    else:
        print("  No ground truth - detection only")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f" Cannot open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f" Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    if output_path:
        print(f"   Output: {output_path}")
    print(f"  Display mode: {'ENABLED' if display else 'DISABLED (headless)'}")
    print("=" * 70)
    print("\n Processing...\n")
    
    if display:
        print("  Window controls: Q to quit, SPACE to pause")
        print("  Look for window named 'Pass Detection'\n")
        cv2.namedWindow('Pass Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pass Detection', width, height)
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track previous frame for velocity
    prev_frame_data = {
        'ball_position': None,
        'players': {},
        'frame_num': 0
    }
    
    frame_buffer = deque(maxlen=sequence_length)
    frame_num = 0
    detections_by_label = {label: 0 for label in labels}
    cooldown_by_label = {label: 0 for label in labels}
    detected_events = []
    
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            for label in labels:
                cooldown_by_label[label] = max(0, cooldown_by_label[label] - 1)
            
            results_players = yolo.track(
                frame, 
                persist=True, 
                tracker='bytetrack.yaml',
                classes=[0], #Players
                conf=0.15,
                iou=0.45,
                max_det=22,
                verbose=False,
                device=device
            )[0]
            
            results_ball = yolo(
                frame,
                classes=[32], #Ball class
                conf=0.02,
                verbose=False,
                device=device
            )[0]
            
            players = []
            if results_players.boxes is not None:
                for box in results_players.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    players.append({
                        'track_id': track_id,
                        'centroid': centroid,
                        'bbox': bbox,
                        'conf': float(box.conf[0])
                    })
            
            ball_pos = None
            ball_conf = 0.0
            
            if results_ball.boxes is not None and len(results_ball.boxes) > 0:
                best_box = max(results_ball.boxes, key=lambda b: float(b.conf[0]))
                bbox = best_box.xyxy[0].cpu().numpy()
                ball_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                ball_conf = float(best_box.conf[0])
            
            # CREATE FEATURES WITH VELOCITY
            features = []
            
            if input_size == 44:  # With velocity
                # BALL (4D: x, y, vx, vy)
                if ball_pos:
                    ball_x = ball_pos[0] / width
                    ball_y = ball_pos[1] / height
                    
                    # Calculate velocity
                    if prev_frame_data['ball_position'] is not None:
                        time_delta = frame_num - prev_frame_data['frame_num']
                        if time_delta > 0:
                            prev_ball = prev_frame_data['ball_position']
                            ball_vx = (ball_pos[0] - prev_ball[0]) / width / time_delta
                            ball_vy = (ball_pos[1] - prev_ball[1]) / height / time_delta
                        else:
                            ball_vx, ball_vy = 0.0, 0.0
                    else:
                        ball_vx, ball_vy = 0.0, 0.0
                    
                    features.extend([ball_x, ball_y, ball_vx, ball_vy])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
                
                # PLAYERS (40D: 10 players × 4D each)
                if ball_pos and players:
                    players_sorted = sorted(
                        players,
                        key=lambda p: np.sqrt(
                            (p['centroid'][0] - ball_pos[0])**2 + 
                            (p['centroid'][1] - ball_pos[1])**2
                        )
                    )
                else:
                    players_sorted = players
                
                for i in range(10):
                    if i < len(players_sorted):
                        p = players_sorted[i]
                        track_id = p['track_id']
                        pos = p['centroid']
                        
                        px = pos[0] / width
                        py = pos[1] / height
                        
                        # Calculate velocity
                        if track_id in prev_frame_data['players']:
                            time_delta = frame_num - prev_frame_data['frame_num']
                            if time_delta > 0:
                                prev_pos = prev_frame_data['players'][track_id]
                                pvx = (pos[0] - prev_pos[0]) / width / time_delta
                                pvy = (pos[1] - prev_pos[1]) / height / time_delta
                            else:
                                pvx, pvy = 0.0, 0.0
                        else:
                            pvx, pvy = 0.0, 0.0
                        
                        features.extend([px, py, pvx, pvy])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
                
                # Update previous frame data
                if ball_pos:
                    prev_frame_data['ball_position'] = ball_pos
                prev_frame_data['players'] = {
                    p['track_id']: p['centroid'] for p in players_sorted
                }
                prev_frame_data['frame_num'] = frame_num
            
            else:  # Without velocity (32D - backward compatible)
                # BALL (2D: x, y)
                if ball_pos:
                    features.extend([ball_pos[0] / width, ball_pos[1] / height])
                else:
                    features.extend([0.0, 0.0])
                
                # PLAYERS (30D: 10 players × 3D each)
                if ball_pos and players:
                    players_sorted = sorted(
                        players,
                        key=lambda p: np.sqrt(
                            (p['centroid'][0] - ball_pos[0])**2 + 
                            (p['centroid'][1] - ball_pos[1])**2
                        )
                    )
                else:
                    players_sorted = players
                
                for i in range(10):
                    if i < len(players_sorted):
                        p = players_sorted[i]
                        features.extend([
                            p['centroid'][0] / width,
                            p['centroid'][1] / height,
                            0.0
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
            
            frame_buffer.append(features)
            
            # Predict
            detected_label = None
            max_confidence = 0.0
            
            if len(frame_buffer) == sequence_length:
                sequence = torch.FloatTensor([list(frame_buffer)]).to(device)
                
                with torch.no_grad():
                    output = model(sequence)
                    probs = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()
                    
                    for i, label in enumerate(labels):
                        confidence = probs[0][i].item()
                        
                        if (predicted_class == i and 
                            confidence >= confidence_threshold and 
                            cooldown_by_label[label] == 0):
                            
                            detected_label = label
                            max_confidence = confidence
                            detections_by_label[label] += 1
                            cooldown_by_label[label] = 40
                            
                            detected_events.append({
                                'frame': frame_num,
                                'label': label,
                                'confidence': confidence,
                                'time': frame_num / fps
                            })
                            break
            
            # Visualize
            vis_frame = frame.copy()
            
            for i, player in enumerate(players_sorted if 'players_sorted' in locals() else []):
                bbox = player['bbox'].astype(int)
                
                color = (0, 255, 0) if i == 0 else (255, 100, 0)
                thickness = 3 if i == 0 else 2
                
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             color, thickness)
            
            if ball_pos:
                ball_int = tuple(map(int, ball_pos))
                cv2.circle(vis_frame, ball_int, 12, (0, 0, 255), -1)
                cv2.circle(vis_frame, ball_int, 14, (255, 255, 255), 2)
            
            if detected_label:
                if detected_label == 'PASS':
                    color = (0, 255, 0)
                elif detected_label == 'DRIVE':
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)
                
                cv2.rectangle(vis_frame, (40, 70), (700, 180), color, -1)
                cv2.rectangle(vis_frame, (40, 70), (700, 180), (255, 255, 255), 3)
                
                cv2.putText(vis_frame, f"{detected_label} DETECTED!", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                cv2.putText(vis_frame, f"Confidence: {max_confidence:.2f}", (50, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Compact stats display
            mode_str = f"{input_size}D {'+ velocity' if input_size == 44 else ''}"
            stats_lines = [
                f"Frame: {frame_num}/{total_frames}",
                f"PASS: {detections_by_label.get('PASS', 0)} | DRIVE: {detections_by_label.get('DRIVE', 0)} | BG: {detections_by_label.get('BACKGROUND', 0)}",
                f"Players: {len(players)} | Ball: {'YES' if ball_pos else 'NO'}"
            ]
            
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, 5), (320, 65), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame)
            
            y = 18
            for stat in stats_lines:
                cv2.putText(vis_frame, stat, (8, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_frame, stat, (8, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20
            
            if writer:
                writer.write(vis_frame)
            
            if display:
                cv2.imshow('Pass Detection', vis_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("\n  Stopped by user")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
            
            if frame_num % 100 == 0:
                elapsed = time.time() - start_time
                fps_rate = frame_num / elapsed if elapsed > 0 else 0
                progress_pct = 100 * frame_num / total_frames
                
                detections_str = ", ".join([f"{l}: {detections_by_label[l]}" for l in labels])
                print(f"Progress: {progress_pct:.1f}% - {detections_str} - Speed: {fps_rate:.1f} fps")
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(" " * 25 + "DETECTION RESULTS")
    print("=" * 70)
    for label in labels:
        print(f"{label} detections: {detections_by_label[label]}")
    print(f"Processing time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Average speed: {frame_num/elapsed_time:.1f} fps")
    
    if detected_events:
        print(f"\nDetected events (first 20):")
        for i, event in enumerate(detected_events[:20], 1):
            minutes = int(event['time'] // 60)
            seconds = int(event['time'] % 60)
            print(f"  {i:2d}. Frame {event['frame']:5d} ({minutes:02d}:{seconds:02d}) - "
                  f"{event['label']:10s} - Conf: {event['confidence']:.3f}")
        
        if len(detected_events) > 20:
            print(f"  ... and {len(detected_events) - 20} more")
    
    if ground_truth_labels:
        print("\n" + "=" * 70)
        print(" " * 25 + "EVALUATION METRICS")
        print("=" * 70)
        
        metrics = compute_detection_metrics(
            detected_events, ground_truth_labels, total_frames, labels
        )
        
        print("\nFRAME-LEVEL METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        
        print("\nPER-CLASS METRICS:")
        for label in label_names:
            print(f"  {label}:")
            print(f"    F1:        {metrics['f1_per_class'][label]:.4f}")
            print(f"    Precision: {metrics['precision_per_class'][label]:.4f}")
            print(f"    Recall:    {metrics['recall_per_class'][label]:.4f}")
        
        print("\nEVENT-LEVEL METRICS:")
        event = metrics['event_level']
        print(f"  Precision: {event['precision']:.4f}")
        print(f"  Recall:    {event['recall']:.4f}")
        print(f"  F1 Score:  {event['f1']:.4f}")
        print(f"  True Positives:  {event['true_positives']}")
        print(f"  False Positives: {event['false_positives']}")
        print(f"  False Negatives: {event['false_negatives']}")
        
        print("\n" + "=" * 70)
        print("SAVING TEST RESULTS")
        print("=" * 70)
        
        cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], label_names, cm_path)
        
        report_path = os.path.join(output_dir, 'test_report.txt')
        save_test_report(metrics, label_names, report_path)
        
        metrics_dict = {
            'frame_level': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro'])
            },
            'event_level': {
                'precision': float(event['precision']),
                'recall': float(event['recall']),
                'f1': float(event['f1']),
                'true_positives': event['true_positives'],
                'false_positives': event['false_positives'],
                'false_negatives': event['false_negatives']
            },
            'per_class': {}
        }
        
        for label in label_names:
            metrics_dict['per_class'][label] = {
                'f1': float(metrics['f1_per_class'][label]),
                'precision': float(metrics['precision_per_class'][label]),
                'recall': float(metrics['recall_per_class'][label])
            }
        
        json_path = os.path.join(output_dir, 'test_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"    Test metrics JSON: {json_path}")
    
    if output_path:
        print(f"\n Output video: {output_path}")
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LSTM with Velocity Support')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--lstm', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--confidence', type=float, default=0.6)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--yolo', type=str, default='yolov8n.pt')
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--label-groups', type=str, default=None)
    parser.add_argument('--group-pass-types', action='store_true')
    parser.add_argument('--ground-truth', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='test_results')
    
    args = parser.parse_args()
    
    override_label_groups = None
    
    if args.group_pass_types:
        override_label_groups = {
            'PASS': ['PASS', 'HIGH PASS', 'HEADER', 'CROSS', 'CLEARANCE']
        }
        print("  Overriding with predefined pass grouping\n")
    elif args.label_groups:
        try:
            override_label_groups = json.loads(args.label_groups)
            print("  Overriding with custom grouping\n")
        except json.JSONDecodeError as e:
            print(f" Error parsing --label-groups: {e}")
            exit(1)
    
    test_pass_detection(
        video_path=args.video,
        model_path=args.lstm,
        output_path=args.output,
        confidence_threshold=args.confidence,
        display=not args.no_display,
        yolo_model=args.yolo,
        labels=args.labels,
        num_classes=args.num_classes,
        override_label_groups=override_label_groups,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir
    )
