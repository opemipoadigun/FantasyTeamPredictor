import subprocess
import sys

scripts = [
    "ingestion.py",
    "preprocessing.py",
    "train.py",
    "predict.py",
    "optimizer.py",
    "chip_advisor.py",
]

for script in scripts:
    print(f"\n--- Running {script} ---")
    subprocess.run([sys.executable, script], check=True)
