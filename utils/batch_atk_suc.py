import os
import subprocess
from pathlib import Path
import pandas as pd

def run_evaluations():
    # Define paths
    det_dir = Path("/home/jiaruili/Documents/exp/advTraj/diffPedModels/parallel_baseline_atk/strongSORT_det")
    gt_dir = Path("/home/jiaruili/Documents/exp/advTraj/diffPedModels/parallel_baseline_atk/bboxes")
    
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
                "utils/atk_success.py",
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
            
            # Initialize default values
            metrics['num_cross_switches'] = 0
            metrics['has_cross_switch'] = 0
            
            for line in output_lines:
                if "Number of cross switches:" in line:
                    metrics['num_cross_switches'] = int(line.split(": ")[1])
                    if metrics['num_cross_switches'] > 0:
                        metrics['has_cross_switch'] = 1

            # Store detailed switch information if available
            switch_details = []
            in_switch_details = False
            for line in output_lines:
                if "Detailed Cross Switches:" in line:
                    in_switch_details = True
                    continue
                if in_switch_details and line.strip():
                    if "Frame" in line:
                        switch_details.append(line)
            
            metrics['switch_details'] = '; '.join(switch_details) if switch_details else "No switches"
                    
            results.append(metrics)
            print(f"Successfully processed {timestamp}")
            print(f"Number of Cross Switches: {metrics['num_cross_switches']}")
            if metrics['has_cross_switch']:
                print("!!!!! CROSS SWITCH DETECTED !!!!!")
                print("Switch details:", metrics['switch_details'])
            
        except Exception as e:
            print(f"Error processing {timestamp}: {str(e)}")
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        print("\nSummary Statistics:")
        print(f"Average Number of Cross Switches: {df['num_cross_switches'].mean():.2f}")
        print(f"Proportion of Sequences with Cross Switches: {df['has_cross_switch'].mean():.4f}")
        
        # Additional analysis
        switches_detected = df[df['has_cross_switch'] == 1]
        if not switches_detected.empty:
            print("\nSequences with Cross Switches:")
            for _, row in switches_detected.iterrows():
                print(f"\nTimestamp: {row['timestamp']}")
                print(f"Number of switches: {row['num_cross_switches']}")
                print(f"Details: {row['switch_details']}")
        
        # Save results to CSV
        output_file = "tracking_cross_switch_evaluation_results.csv"
        # Don't save the details column to CSV to keep it clean
        df_to_save = df.drop('switch_details', axis=1)
        df_to_save.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
        
        # Save detailed results including switch information to a separate file
        detailed_output = "tracking_cross_switch_detailed_results.txt"
        with open(detailed_output, 'w') as f:
            for _, row in df.iterrows():
                f.write(f"\nTimestamp: {row['timestamp']}\n")
                f.write(f"Number of cross switches: {row['num_cross_switches']}\n")
                f.write(f"Switch details: {row['switch_details']}\n")
                f.write("-" * 50 + "\n")
        print(f"Detailed switch information saved to {detailed_output}")
    else:
        print("No results to summarize")

if __name__ == "__main__":
    run_evaluations()