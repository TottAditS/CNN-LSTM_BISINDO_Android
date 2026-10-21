"""
Settings shared by both models and by Training/Scripts/comparator.py.

Each model writes its *_evaluation_summary.json to RESULTS_DIR, where the
comparator reads it. Model-specific settings live in each model's Config/.
"""

from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Datasets").exists() and (parent / "PreProcessing").exists():
            return parent
    raise RuntimeError(
        "Project root not found. "
        "The root must contain the Datasets/ and PreProcessing/ folders."
    )


ROOT = find_project_root()

TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output"
RESULTS_DIR = TRAINING_OUTPUT_DIR / "results"
CHARTS_DIR = TRAINING_OUTPUT_DIR / "charts"


def create_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
