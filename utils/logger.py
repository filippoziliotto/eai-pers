# logger.py
import logging
import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_filename: str = "run.log", log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{timestamp}_{log_filename}")

        self.logger = logging.getLogger("UnifiedLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        # Redirect all `print()` to logger
        sys.stdout = self

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.info(message)

    def flush(self):
        pass

    def info(self, message):
        self.logger.info(message)
