import os
import subprocess
from pathlib import Path
import pandas as pd

def run_evaluations():
    # Define paths
    det_dir = Path("/home/jiaruili/Documents/exp/advTraj/baselines/parallel_baseline_atk/strongSORT_det")
    gt_dir = Path("/home/jiaruili/Documents/exp/advTraj/baselines/parallel_baseline_atk/bboxes")
    
    # Get all detection files
    det_files = list(det_dir.glob("*.txt"))
    
    # Store results
    results = []
    
    for det_file in det_files:
        # Extract timestamp from filename
        timestamp = det_file.stem  # Gets filename without extension
        gt_file = gt_dir / f"{timestamp}.npy"
        
        # Check if corresponding ground truth file exists
        if not gt_file.exists():
            print(f"Warning: No matching ground truth file for {det_file}")
            continue
            
        print(f"\nProcessing {timestamp}")
        print(f"Detection file: {det_file}")
        print(f"Ground truth file: {gt_file}")
        
        # Run evaluation script
        try:
            cmd = [
                "python",
                "utils/atk_success.py",  # Replace with your evaluation script name
                "--gt-path", str(gt_file),
                "--det-path", str(det_file),
                "--iou-threshold", "0.5"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error processing {timestamp}")
                print("Error output:", result.stderr)
                continue
                
            # Parse the output to extract metrics
            output_lines = result.stdout.split('\n')
            metrics = {}
            metrics['timestamp'] = timestamp
            
            for line in output_lines:
                if "ID Switches:" in line:
                    metrics['id_switches'] = int(line.split(": ")[1])
                elif "ID Precision:" in line:
                    metrics['id_precision'] = float(line.split(": ")[1])
                elif "Persistent Switches:" in line:
                    metrics['persistent_switches'] = int(line.split(": ")[1])
                elif "Total Matches:" in line:
                    metrics['total_matches'] = int(line.split(": ")[1])

            if metrics['persistent_switches'] > 0:
                metrics['cnt_persistent_switches'] = 1
            else:
                metrics['cnt_persistent_switches'] = 0
                    
            results.append(metrics)
            print(f"Successfully processed {timestamp}")
            print(f"ID Switches: {metrics['id_switches']}")
            print(f"ID Precision: {metrics['id_precision']:.4f}")
            print(f"Persistent Switches: {metrics['persistent_switches']}")
            if metrics['persistent_switches'] > 0:
                print("!!!!!")
            
        except Exception as e:
            print(f"Error processing {timestamp}: {str(e)}")
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate average metrics
        print("\nSummary Statistics:")
        print(f"Average ID Switches: {df['id_switches'].mean():.2f}")
        print(f"Average ID Precision: {df['id_precision'].mean():.4f}")
        print(f"Average Persistent Switches: {df['persistent_switches'].mean():.2f}")
        print(f"Average Count of Persistent Switches: {df['cnt_persistent_switches'].mean():.5f}")
        
        # Save results to CSV
        output_file = "tracking_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
    else:
        print("No results to summarize")

if __name__ == "__main__":
    run_evaluations()