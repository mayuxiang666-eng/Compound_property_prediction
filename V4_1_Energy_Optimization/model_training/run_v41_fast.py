import os
import sys

# Set unbuffered
sys.stdout.reconfigure(line_buffering=True)

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from model_training.run_v41_energy_experiment import run_v41_master_experiment

if __name__ == '__main__':
    run_v41_master_experiment()
