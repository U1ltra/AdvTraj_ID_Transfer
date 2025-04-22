import os
import subprocess
from pathlib import Path
import pandas as pd
import json

def parse_switch_details(output_lines):
    """Parse switch details from output lines and return structured information."""
    oneway_switches = []
    twoway_switches = []
    current_section = None
    
    for line in output_lines:
        if "One-way Switch Details:" in line:
            current_section = "oneway"
            continue
        elif "Two-way Switch Details:" in line:
            current_section = "twoway"
            continue
            
        if current_section and "Frame" in line:
            # Parse frame range and duration
            frame_info = line.split("(Duration:")[0]
            start_frame = int(frame_info.split("Frame ")[1].split(" to ")[0])
            end_frame = int(frame_info.split(" to ")[1])
            duration = int(line.split("Duration: ")[1].split(" frames")[0])
            
            # Parse ID information from next lines
            switch_info = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': duration
            }
            
            if current_section == "oneway":
                oneway_switches.append(switch_info)
            else:
                twoway_switches.append(switch_info)
    
    return oneway_switches, twoway_switches

def calculate_switch_statistics(switches):
    """Calculate statistics for a list of switches."""
    if not switches:
        return {
            'count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'max_duration': 0,
            'min_duration': 0
        }
    
    durations = [switch['duration'] for switch in switches]
    return {
        'count': len(switches),
        'total_duration': sum(durations),
        'avg_duration': sum(durations) / len(durations),
        'max_duration': max(durations),
        'min_duration': min(durations)
    }

def run_evaluations():
    # Define paths
    det_dir = Path("/home/jiaruili/Documents/exp/advTraj/diffPedModels/parallel_baseline_atk/strongSORT_det")
    gt_dir = Path("/home/jiaruili/Documents/exp/advTraj/diffPedModels/parallel_baseline_atk/bboxes")
    
    # Get all detection files
    det_files = list(det_dir.glob("*.txt"))
    
    # Store results
    results = []
    
    for det_file in det_files[:]:
        # Extract timestamp from filename
        timestamp = det_file.stem
        gt_file = gt_dir / f"{timestamp}.npy"
        
        # Check if corresponding ground truth file exists
        if not gt_file.exists():
            print(f"Warning: No matching ground truth file for {det_file}")
            continue
            
        print(f"\nProcessing {timestamp}")
        print(f"Detection file: {det_file}")
        print(f"Ground truth file: {gt_file}")
        
        try:
            cmd = [
                "python",
                "utils/atk_success.py",
                "--gt-path", str(gt_file),
                "--det-path", str(det_file),
                "--iou-threshold", "0.3"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error processing {timestamp}")
                print("Error output:", result.stderr)
                continue
            
            output_lines = result.stdout.split('\n')
            
            # Parse switch details
            oneway_switches, twoway_switches = parse_switch_details(output_lines)
            
            # Calculate statistics
            oneway_stats = calculate_switch_statistics(oneway_switches)
            twoway_stats = calculate_switch_statistics(twoway_switches)
            
            # Create metrics dictionary
            metrics = {
                'timestamp': timestamp,
                # One-way switch metrics
                'num_oneway_switches': oneway_stats['count'],
                'has_oneway_switch': 1 if oneway_stats['count'] > 0 else 0,
                'oneway_total_duration': oneway_stats['total_duration'],
                'oneway_avg_duration': round(oneway_stats['avg_duration'], 2),
                'oneway_max_duration': oneway_stats['max_duration'],
                'oneway_min_duration': oneway_stats['min_duration'],
                # Two-way switch metrics
                'num_twoway_switches': twoway_stats['count'],
                'has_twoway_switch': 1 if twoway_stats['count'] > 0 else 0,
                'twoway_total_duration': twoway_stats['total_duration'],
                'twoway_avg_duration': round(twoway_stats['avg_duration'], 2),
                'twoway_max_duration': twoway_stats['max_duration'],
                'twoway_min_duration': twoway_stats['min_duration'],
                # Store detailed switch information
                'oneway_switches': oneway_switches,
                'twoway_switches': twoway_switches
            }
            
            results.append(metrics)
            
            # Print summary for this sequence
            print(f"\nResults for {timestamp}:")
            print("One-way Switches:")
            print(f"  Count: {metrics['num_oneway_switches']}")
            if metrics['has_oneway_switch']:
                print(f"  Average Duration: {metrics['oneway_avg_duration']} frames")
                print(f"  Max Duration: {metrics['oneway_max_duration']} frames")
                print("  !!!!! ONE-WAY SWITCH DETECTED !!!!!")
            
            print("\nTwo-way Switches:")
            print(f"  Count: {metrics['num_twoway_switches']}")
            if metrics['has_twoway_switch']:
                print(f"  Average Duration: {metrics['twoway_avg_duration']} frames")
                print(f"  Max Duration: {metrics['twoway_max_duration']} frames")
                print("  !!!!! TWO-WAY SWITCH DETECTED !!!!!")
            
        except Exception as e:
            print(f"Error processing {timestamp}: {str(e)}")
    
    # Create summary DataFrame
    if results:
        # Create DataFrame without the detailed switch lists
        df_metrics = [{k: v for k, v in r.items() if k not in ['oneway_switches', 'twoway_switches']} 
                     for r in results]
        df = pd.DataFrame(df_metrics)
        
        # Calculate summary statistics
        print("\nSummary Statistics:")
        print("\nOne-way Switch Statistics:")
        print(f"Total Sequences with Switches: {df['has_oneway_switch'].sum()}")
        print(f"Proportion of Sequences with Switches: {df['has_oneway_switch'].mean():.4f}")
        print(f"Average Number of Switches per Sequence: {df['num_oneway_switches'].mean():.2f}")
        print(f"Average Duration of Switches: {df['oneway_avg_duration'].mean():.2f} frames")
        print(f"Maximum Switch Duration: {df['oneway_max_duration'].max()} frames")
        
        print("\nTwo-way Switch Statistics:")
        print(f"Total Sequences with Switches: {df['has_twoway_switch'].sum()}")
        print(f"Proportion of Sequences with Switches: {df['has_twoway_switch'].mean():.4f}")
        print(f"Average Number of Switches per Sequence: {df['num_twoway_switches'].mean():.2f}")
        print(f"Average Duration of Switches: {df['twoway_avg_duration'].mean():.2f} frames")
        print(f"Maximum Switch Duration: {df['twoway_max_duration'].max()} frames")
        
        # Save metrics to CSV
        output_file = "tracking_cross_switch_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nMetrics saved to {output_file}")
        
        # Save detailed results including switch information
        detailed_output = "tracking_id_switch_detailed_results.txt"
        with open(detailed_output, 'w') as f:
            for result in results:
                f.write(f"\nTimestamp: {result['timestamp']}\n")
                
                f.write("\nOne-way Switches:\n")
                for i, switch in enumerate(result['oneway_switches'], 1):
                    f.write(f"Switch {i}:\n")
                    f.write(f"  Frames: {switch['start_frame']} to {switch['end_frame']}\n")
                    f.write(f"  Duration: {switch['duration']} frames\n")
                
                f.write("\nTwo-way Switches:\n")
                for i, switch in enumerate(result['twoway_switches'], 1):
                    f.write(f"Switch {i}:\n")
                    f.write(f"  Frames: {switch['start_frame']} to {switch['end_frame']}\n")
                    f.write(f"  Duration: {switch['duration']} frames\n")
                
                f.write("-" * 50 + "\n")
        
        print(f"Detailed switch information saved to {detailed_output}")
    else:
        print("No results to summarize")

if __name__ == "__main__":
    run_evaluations()