import subprocess
import sys

def run_command(cmd_args):
    """
    Runs the python analysis script with the given arguments.
    """
    # Prepend the current python executable to ensure we use the same environment
    cmd = [sys.executable] + cmd_args
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        # We'll continue to the next one even if one fails, to try and get as much done as possible.
        # But we'll print a clear error.
        print("Continuing to next experiment...\n")
    else:
        print("Analysis finished successfully.\n")

def main():
    base_script = "7_Analysis/python_scripts/analyze_gaze.py"
    
    experiments = [
        {
            "config": "4_Experiments/configs/gaze_earlyfusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_earlyfusion/add/best_model.pt",
            "model_type": "early",
            "exp_name": "Early_(Add)"
        },
        {
            "config": "4_Experiments/configs/gaze_earlyfusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_earlyfusion/concate/best_model.pt", # Note: 'concate' as per path structure
            "model_type": "early",
            "exp_name": "Early_(Concat)"
        },
        {
            "config": "4_Experiments/configs/gaze_earlyfusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_earlyfusion/multiply/best_model.pt",
            "model_type": "early",
            "exp_name": "Early_(Multiply)"
        },
        {
            "config": "4_Experiments/configs/gaze_earlyfusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_earlyfusion/subtract_abs/best_model.pt",
            "model_type": "early",
            "exp_name": "Early_(Subtract)"
        },
        {
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_latefusion/add/best_model.pt",
            "model_type": "late",
            "exp_name": "Late_Fusion_(Add)"
        },
        {
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_latefusion/concat/best_model.pt",
            "model_type": "late",
            "exp_name": "Late_Fusion_(Concat)"
        },
        {
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_latefusion/full/best_model.pt",
            "model_type": "late",
            "exp_name": "Late_Fusion_(Full)"
        },
        {
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_latefusion/multiply/best_model.pt",
            "model_type": "late",
            "exp_name": "Late_Fusion_(Multiply)"
        },
        {
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "checkpoint": "4_Experiments/runs/gaze_latefusion/subtract/best_model.pt",
            "model_type": "late",
            "exp_name": "Late_Fusion_(Subtract)"
        }
    ]

    print(f"Starting batch of {len(experiments)} analysis tasks...")
    print("="*60)

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] Analyzing {exp['exp_name']}...")
        
        args = [
            base_script,
            "--config", exp['config'],
            "--checkpoint", exp['checkpoint'],
            "--model_type", exp['model_type'],
            "--exp_name", exp['exp_name']
        ]
        
        run_command(args)

    print("All analysis tasks completed.")

if __name__ == "__main__":
    main()
