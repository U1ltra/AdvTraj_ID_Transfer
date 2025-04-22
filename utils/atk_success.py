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

def evaluate_cross_switches(gt_data, det_data, iou_threshold=0.5):
    """
    Evaluate ID switches between two objects, including switch duration
    gt_data: numpy array of shape (4, num_frames, 4) containing ground truth boxes. 
             [0, :, :] for victim, [1, :, :] for attacker, 
             [2, :, :] for victim's predicted kalman filter bbx, 
             [3, :, :] for attacker's predicted kalman filter bbx
    det_data: pandas DataFrame containing detection results
    iou_threshold: IoU threshold for matching detections
    """
    num_objects, num_frames, _ = gt_data.shape
    num_objects = 2
    assert num_objects == 2, "This function is designed for exactly 2 objects"
    
    # Dictionary to store the tracking history for each GT object
    gt_tracking_history = {
        0: [],  # List of (frame, track_id) for GT object 0
        1: []   # List of (frame, track_id) for GT object 1
    }

    # Process frame by frame
    for frame in range(num_frames):
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
            if cost_matrix[gt_idx, det_idx] < 0:  # valid match
                track_id = frame_dets.iloc[det_idx]['id']
                gt_tracking_history[gt_idx].append((frame, track_id))
            # print(f"Frame {frame}: GT{gt_idx} matched with ID {frame_dets.iloc[det_idx]['id']}")
            
    
    # Process frames where both objects are detected
    valid_frames = []
    for frame in range(num_frames):
        frame0 = next(((f, tid) for f, tid in gt_tracking_history[0] if f == frame), None)
        frame1 = next(((f, tid) for f, tid in gt_tracking_history[1] if f == frame), None)
        
        if frame0 and frame1:  # Both objects detected
            valid_frames.append((frame, frame0[1], frame1[1]))
    
    if not valid_frames:
        return {
            'oneway_switches': [],
            'twoway_switches': [],
            'num_oneway_switches': 0,
            'num_twoway_switches': 0
        }

    # Initialize switch tracking
    oneway_switches = []
    twoway_switches = []
    current_oneway_switch = None
    current_twoway_switch = None

    # Initialize current assignment from first valid frame
    first_frame = valid_frames[0]
    current_assignment = {
        0: first_frame[1],  # GT object 0's initial track ID
        1: first_frame[2]   # GT object 1's initial track ID
    }
    initial_assignment = current_assignment.copy()

    # Look for switches and track their duration
    for i, (frame, id0, id1) in enumerate(valid_frames[1:]):
        is_twoway_switch = (id0 == current_assignment[1] and id1 == current_assignment[0])
        is_oneway_switch = (id0 == current_assignment[1] and id1 != current_assignment[0]) # victim has the attacker's ID
        
        # Handle two-way switches
        if is_twoway_switch:
            if current_twoway_switch is None:
                # Start new two-way switch
                current_twoway_switch = {
                    'start_frame': frame,
                    'before': current_assignment.copy(),
                    'switched_ids': {0: id0, 1: id1},
                    'duration': 1
                }
            else:
                # Different switch pattern, end current switch
                current_twoway_switch['end_frame'] = frame - 1
                twoway_switches.append(current_twoway_switch)
                # Start new switch
                current_twoway_switch = {
                    'start_frame': frame,
                    'before': current_assignment.copy(),
                    'switched_ids': {0: id0, 1: id1},
                    'duration': 1
                }
        elif current_twoway_switch is not None:
            # if the existing switch IDs matches exactly, continue the switch
            if id0 == current_twoway_switch['switched_ids'][0] and id1 == current_twoway_switch['switched_ids'][1]:
                current_twoway_switch['duration'] += 1
            else:
                # End current two-way switch
                current_twoway_switch['end_frame'] = frame - 1
                twoway_switches.append(current_twoway_switch)
                current_twoway_switch = None

        # Handle one-way switches
        if is_oneway_switch:
            if current_oneway_switch is None:
                # Start new one-way switch
                current_oneway_switch = {
                    'start_frame': frame,
                    'before': current_assignment.copy(),
                    'switched_ids': {0: id0, 1: id1},
                    'duration': 1
                }
            else:
                # Different switch pattern, end current switch
                current_oneway_switch['end_frame'] = frame - 1
                oneway_switches.append(current_oneway_switch)
                # Start new switch
                current_oneway_switch = {
                    'start_frame': frame,
                    'before': current_assignment.copy(),
                    'switched_ids': {0: id0, 1: id1},
                    'duration': 1
                }
        elif current_oneway_switch is not None:
            # if the existing switch IDs matches exactly, continue the switch
            if id0 == current_oneway_switch['switched_ids'][0] and id1 == current_oneway_switch['switched_ids'][1]:
                current_oneway_switch['duration'] += 1 # increment duration for each persisted frame (not necessarily consecutive)
            else:
                # End current one-way switch
                current_oneway_switch['end_frame'] = frame - 1 # since the det can be missing for a number of frames
                oneway_switches.append(current_oneway_switch) # this will record the frame at which the a new det is found and the id is switched again
                current_oneway_switch = None

        # Update current assignment
        current_assignment = {0: id0, 1: id1}

    # Handle any ongoing switch at the end of the sequence
    last_frame = valid_frames[-1][0]
    if current_oneway_switch is not None:
        current_oneway_switch['end_frame'] = last_frame
        oneway_switches.append(current_oneway_switch)
    if current_twoway_switch is not None:
        current_twoway_switch['end_frame'] = last_frame
        twoway_switches.append(current_twoway_switch)
    
    return {
        'oneway_switches': oneway_switches,
        'num_oneway_switches': len(oneway_switches),
        'twoway_switches': twoway_switches,
        'num_twoway_switches': len(twoway_switches),
        'tracking_history': gt_tracking_history
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multi-object tracking ID switches')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Path to ground truth numpy file')
    parser.add_argument('--det-path', type=str, required=True,
                        help='Path to detection results text file')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
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
    
    # Evaluate cross switches
    cross_results = evaluate_cross_switches(gt_data, det_data, args.iou_threshold)
    
    print("\nCross-Object ID Switch Analysis:")
    print(f"Number of one-way switches: {cross_results['num_oneway_switches']}")
    print(f"Number of two-way switches: {cross_results['num_twoway_switches']}")
    
    if cross_results['num_oneway_switches'] > 0:
        print("\nOne-way Switch Details:")
        for switch in cross_results['oneway_switches']:
            print(f"Frame {switch['start_frame']} to {switch['end_frame']} (Duration: {switch['duration']} frames):")
            print(f"  Before - GT0: {switch['before'][0]}, GT1: {switch['before'][1]}")
            print(f"  After  - GT0: {switch['switched_ids'][0]}, GT1: {switch['switched_ids'][1]}")
    
    if cross_results['num_twoway_switches'] > 0:
        print("\nTwo-way Switch Details:")
        for switch in cross_results['twoway_switches']:
            print(f"Frame {switch['start_frame']} to {switch['end_frame']} (Duration: {switch['duration']} frames):")
            print(f"  Before - GT0: {switch['before'][0]}, GT1: {switch['before'][1]}")
            print(f"  After  - GT0: {switch['switched_ids'][0]}, GT1: {switch['switched_ids'][1]}")