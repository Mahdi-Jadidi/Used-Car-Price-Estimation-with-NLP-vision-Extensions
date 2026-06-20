import subprocess
import sys

def run_script(script_path):
    print(f"Running: {script_path}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error occurred while executing {script_path}")
        sys.exit(result.returncode)
    print(f"Successfully finished: {script_path}\n")

if __name__ == "__main__":
    run_script("scripts/load_data.py")
    run_script("scripts/preprocess.py")
    run_script("scripts/feature_engineering.py")
    
    print("Entire AI Data Pipeline executed successfully!")
