from pathlib import Path


#   Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "Inputs"
OUTPUT_DIR = BASE_DIR / "Outputs"
SCRIPTS_DIR = BASE_DIR / "Scripts"

#   Model
TEST_SIZE = 0.8
TARGET = "Class"
RANDOM_STATE = 42   #   Classic!