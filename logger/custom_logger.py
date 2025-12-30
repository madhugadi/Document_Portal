import logging
import os
from datetime import datetime


class CustomLogger:
    def __init__(self, log_dir='logs'):

        # Ensure logs directory exists
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create timestamped log file name
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

        # Configure logging (FIXED)
        logging.basicConfig(
            filename=self.log_file_path,
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s - %(message)s',
        )

    def get_logger(self, name=__file__):
        return logging.getLogger(os.path.basename(name).upper())


# ✅ This must be OUTSIDE the class
if __name__ == "__main__":
    logger_obj = CustomLogger()
    logger = logger_obj.get_logger(__file__)
    logger.info("Custom logger initialized.")
