import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "PathSnap API"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Data directories
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    STUDENT_TEST_DIR: Path = RAW_DATA_DIR / "student_test"
    EVALUATOR_TEST_DIR: Path = RAW_DATA_DIR / "evaluator_test"
    
    # Model & Decoding Artifacts
    MODEL_PATH: Path = BASE_DIR / "models" / "saved" / "best_model.joblib"
    HMM_CONFIG_PATH: Path = BASE_DIR / "hmm" / "config.yaml"
    
    # Logging
    LOG_DIR: Path = BASE_DIR / "api" / "logs"
    LOG_FILE: Path = LOG_DIR / "requests.log"

    # Known 10 student testing trip IDs
    DEFAULT_TEST_TRIPS: list[str] = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]

settings = Settings()
os.makedirs(settings.LOG_DIR, exist_ok=True)
