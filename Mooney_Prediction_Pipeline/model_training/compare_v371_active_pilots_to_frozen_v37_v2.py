"""Compatibility entrypoint for the legacy frozen V3.7 pickle module path."""

from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from Mooney_Prediction_Pipeline.model_training.compare_v371_active_pilots_to_frozen_v37 import main


if __name__ == "__main__":
    main()
