import numpy as np
from collections import defaultdict
import pandas as pd
from scipy.optimize import linear_sum_assignment
import argparse

def load_detection_results(det_file):
    """Load detection results from txt file"""
    # Read detection file: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    det_data = pd.read_csv(det_file, header=None)
    det_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 
                       'conf', 'x', 'y', 'z']
    return det_data

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format"""
    # Convert to [x1, y1, x2, y2] format
    # print(box1, box2)
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[2], box1[3]
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

class IDSwitchInfo:
    def __init__(self, frame, old_id, new_id):
        self.start_frame = frame
        self.old_id = old_id
        self.new_id = new_id
        self.duration = 1
        self.is_continuing = True
        self.end_frame = frame

def evaluate_tracking(gt_data, det_data, iou_threshold=0.5):
    """
    Evaluate tracking results using Hungarian algorithm for optimal matching
    gt_data: numpy array of shape (num_objects, num_frames, 4) containing ground truth boxes
    det_data: pandas DataFrame containing detection results
    """
    num_objects, num_frames, _ = gt_data.shape
    id_switches = 0
    correct_ids = 0
    total_matches = 0
    
    # Dictionary to store the first assigned tracking ID for each ground truth object
    gt_to_track_id = {}  # gt_idx -> first_assigned_track_id
    
    # List to store switch information for each GT object
    switch_history = [[] for _ in range(num_objects)]  # List of IDSwitchInfo objects for each GT object
    
    # Process frame by frame
    for frame in range(num_frames):
        # Get detections for current frame
        frame_dets = det_data[det_data['frame'] == frame].reset_index(drop=True)
        
        if len(frame_dets) == 0:
            continue
            
        # Create cost matrix for Hungarian algorithm
        cost_matrix = np.zeros((num_objects, len(frame_dets)))
        
        # Fill cost matrix with negative IoU (since Hungarian minimizes cost)
        for i in range(num_objects):
            gt_box = gt_data[i, frame]
            for j, det in frame_dets.iterrows():
                det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
                iou = calculate_iou(gt_box, det_box)
                # Set high cost for low IoU matches
                cost_matrix[i, frame_dets.index[frame_dets.index == j][0]] = -iou if iou > iou_threshold else 1000

        # Find optimal assignment
        gt_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Process matches
        for gt_idx, det_idx in zip(gt_indices, det_indices):
            # Only count if IoU is above threshold
            if cost_matrix[gt_idx, det_idx] < 0:  # negative cost means valid IoU
                total_matches += 1
                det = frame_dets.iloc[det_idx]
                track_id = det['id']
                
                # If this is the first time we see this ground truth object
                if gt_idx not in gt_to_track_id:
                    gt_to_track_id[gt_idx] = track_id
                    correct_ids += 1
                else:
                    # Check if the tracking ID is consistent with the first assignment
                    if gt_to_track_id[gt_idx] == track_id:
                        correct_ids += 1
                        # If there was an ongoing switch, mark it as ended
                        if switch_history[gt_idx] and switch_history[gt_idx][-1].is_continuing:
                            switch_history[gt_idx][-1].is_continuing = False
                            switch_history[gt_idx][-1].end_frame = frame - 1
                    else:
                        id_switches += 1
                        # Check if this is a continuation of the previous switch
                        if (switch_history[gt_idx] and 
                            switch_history[gt_idx][-1].is_continuing and 
                            switch_history[gt_idx][-1].new_id == track_id):
                            # Increment duration of the current switch
                            switch_history[gt_idx][-1].duration += 1
                            switch_history[gt_idx][-1].end_frame = frame
                        else:
                            # Record new switch
                            switch_info = IDSwitchInfo(frame, gt_to_track_id[gt_idx], track_id)
                            switch_history[gt_idx].append(switch_info)
    
    # Calculate persistence metrics
    persistent_switches = 0
    avg_switch_duration = 0
    switch_durations = []
    
    for gt_idx in range(num_objects):
        for switch in switch_history[gt_idx]:
            if switch.duration > 1:  # Consider a switch persistent if it lasts more than 1 frame
                persistent_switches += 1
            switch_durations.append(switch.duration)
    
    avg_switch_duration = np.mean(switch_durations) if switch_durations else 0
    
    # Calculate metrics
    id_precision = correct_ids / total_matches if total_matches > 0 else 0
    
    return {
        'id_switches': id_switches,
        'persistent_switches': persistent_switches,
        'avg_switch_duration': avg_switch_duration,
        'correct_ids': correct_ids,
        'total_matches': total_matches,
        'id_precision': id_precision,
        'gt_to_track_id': gt_to_track_id,
        'switch_history': switch_history
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multi-object tracking ID switches')
    parser.add_argument('--gt-path', type=str, required=True,
                        default='~/Documents/exp/advTraj/baselines/surveillance_camera/bboxes/2024_12_19_18_12_42.npy',
                        help='Path to ground truth numpy file')
    parser.add_argument('--det-path', type=str, required=True,
                        default='~/Documents/exp/StrongSORT/surv/2024_12_19_18_12_42.txt',
                        help='Path to detection results text file')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    try:
        gt_data = np.load(args.gt_path)
        print(f"Loaded ground truth data with shape: {gt_data.shape}")
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        exit(1)
        
    try:
        det_data = load_detection_results(args.det_path)
        print(f"Loaded detection data with {len(det_data)} detections")
    except Exception as e:
        print(f"Error loading detection file: {e}")
        exit(1)
    
    # Evaluate
    results = evaluate_tracking(gt_data, det_data, args.iou_threshold)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"ID Switches: {results['id_switches']}")
    print(f"Persistent Switches: {results['persistent_switches']}")
    print(f"Average Switch Duration: {results['avg_switch_duration']:.2f} frames")
    print(f"Correct IDs: {results['correct_ids']}")
    print(f"Total Matches: {results['total_matches']}")
    print(f"ID Precision: {results['id_precision']:.4f}")
    
    # Print detailed switch history
    print("\nDetailed Switch History:")
    for gt_idx, switches in enumerate(results['switch_history']):
        if switches:
            print(f"\nGT Object {gt_idx}:")
            for switch in switches:
                persistence = "Continuing" if switch.is_continuing else f"Ended at frame {switch.end_frame}"
                print(f"  Frame {switch.start_frame}: {switch.old_id} -> {switch.new_id} "
                      f"(Duration: {switch.duration} frames, {persistence})")