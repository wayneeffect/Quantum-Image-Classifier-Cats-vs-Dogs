import os
from pathlib import Path

# If you want to use python-dotenv locally, uncomment the next two lines:
# from dotenv import load_dotenv
# load_dotenv()

class Config:
    """Manages application configurations derived from environment variables."""
    ORACLE_API_URL: str = os.getenv(
        "ORACLE_API_URL", 
        "https://grok-wayne-s-quantum-algorithm.onrender.com/hybrid_vqe_qaoa"
    )
    # Render free tier apps spin down; a 30s timeout accounts for potential cold starts
    ORACLE_TIMEOUT: int = int(os.getenv("ORACLE_TIMEOUT_SECONDS", "30"))
    MATRIX_SIZE: int = int(os.getenv("MATRIX_SIZE", "4"))
    THRESHOLD: float = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.0"))

    @classmethod
    def validate(cls) -> None:
        """Validates that critical configurations are present and valid."""
        if not cls.ORACLE_API_URL.startswith("http"):
            raise ValueError(f"Invalid ORACLE_API_URL protocol: {cls.ORACLE_API_URL}")
        if cls.MATRIX_SIZE not in [2, 4, 8]:
            raise ValueError("MATRIX_SIZE must match a valid qubit mapping scale (2, 4, or 8).")
