import logging
from datetime import datetime
from pathlib import Path


log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logging.basicConfig(
    filename=str(log_file),
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)