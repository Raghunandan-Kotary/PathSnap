import logging
import sys
from datetime import datetime
from api.config import settings

def setup_logger():
    logger = logging.getLogger("PathSnapAPI")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [PathSnap] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler (prints to CMD for evaluator inspection)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler (persists audit trail)
        file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

api_logger = setup_logger()

def log_evaluation_request(trip_id: str, client_ip: str, latency_ms: float, accuracy: float, continuity: float, mode: str = "unknown"):
    api_logger.info(
        f"EVALUATOR_REQ | Trip: {trip_id} | Client: {client_ip} | Mode: {mode} | "
        f"Latency: {latency_ms:.2f}ms | Top-1 Acc: {accuracy * 100:.2f}% | Continuity: {continuity * 100:.2f}%"
    )
