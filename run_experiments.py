import os
import re
import subprocess
import sys

def update_config(config_path, fusion_mode):
    """
    Updates the fusion_mode in the specified YAML config file using regex to preserve comments.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find fusion_mode: "..." or fusion_mode: '...' or fusion_mode: ...
    # This pattern assumes fusion_mode is on its own line or at least clearly defined.
    # Group 1: 'fusion_mode: ' (and whitespace)
    # Group 2: Opening quote (optional)
    # Group 3: The value (until closing quote or boundary)
    # Group 4: Closing quote (optional)
    pattern = r'(fusion_mode:\s*)(["\']?)([^"\\]+\b)(["\']?)'
    
    # Replacement string: Group 1 content + " + new_mode + "
    # We use r'\g<1>' to refer to group 1 safely.
    replacement = r'\g<1>"' + fusion_mode + '"'
    
    new_content = re.sub(pattern, replacement, content)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated {config_path} to fusion_mode: \"{fusion_mode}\"")

def run_experiment(script_path, config_path):
    """
    Runs the python training script with the given config.
    """
    cmd = [sys.executable, script_path, "--config", config_path]
    print(f"Running command: {' '.join(cmd)}")
    
    # running the command and waiting for it to complete
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error running experiment: {cmd}")
        # We might want to stop if one fails, or continue. 
        # For now, let's raise an error to stop execution so the user notices.
        raise RuntimeError(f"Experiment failed with return code {result.returncode}")
    else:
        print("Experiment finished successfully.\n")

def main():
    experiments = [
        {
            "description": "1. Early Fusion - Concat",
            "config": "4_Experiments/configs/gaze_earlyfusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_earlyfusion.py",
            "mode": "concat"
        },
        {
            "description": "2. Late Fusion - Full",
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_latefusion.py",
            "mode": "full"
        },
        {
            "description": "3. Late Fusion - Concat",
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_latefusion.py",
            "mode": "concat"
        },
        {
            "description": "4. Late Fusion - Add",
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_latefusion.py",
            "mode": "add"
        },
        {
            "description": "5. Late Fusion - Subtract",
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_latefusion.py",
            "mode": "subtract"
        },
        {
            "description": "6. Late Fusion - Multiply",
            "config": "4_Experiments/configs/gaze_latefusion.yaml",
            "script": "4_Experiments/scripts/train_gaze_latefusion.py",
            "mode": "multiply"
        }
    ]

    print(f"Starting batch of {len(experiments)} experiments...")
    print("="*60)

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {exp['description']}")
        
        # verify paths exist
        if not os.path.exists(exp['config']):
            print(f"Config file not found: {exp['config']}")
            return
        if not os.path.exists(exp['script']):
            print(f"Script file not found: {exp['script']}")
            return

        try:
            update_config(exp['config'], exp['mode'])
            run_experiment(exp['script'], exp['config'])
        except Exception as e:
            print(f"Failed during experiment {i}: {e}")
            break

    print("All experiments completed.")

if __name__ == "__main__":
    main()