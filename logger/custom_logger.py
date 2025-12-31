import logging
import os
from datetime import datetime


class CustomLogger:
    """
    Centralized logging utility for the application.

    Responsibilities:
    - Create a logs directory if it does not exist
    - Generate a timestamped log file
    - Log messages to BOTH file and terminal
    - Prevent duplicate handlers and propagation issues
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logger configuration.

        :param log_dir: Directory where log files will be stored
        """

        # Absolute path to logs directory
        self.logs_dir = os.path.join(os.getcwd(), log_dir)

        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create timestamp-based log file name
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        """
        Returns a configured logger instance for a given module.

        :param name: Usually __file__ of the calling module
        :return: Configured logger instance
        """

        # Use filename as logger name
        logger_name = os.path.basename(name).upper()
        logger = logging.getLogger(logger_name)

        # Set log level
        logger.setLevel(logging.INFO)

        # Attach handlers only once
        if not logger.handlers:

            # File handler → writes logs to file
            file_handler = logging.FileHandler(
                self.log_file_path, encoding="utf-8"
            )

            # Stream handler → prints logs to terminal
            stream_handler = logging.StreamHandler()

            # Common log format
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s"
            )

            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        # Prevent duplicate logs via root logger
        logger.propagate = False

        return logger


# Standalone test block
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("Custom logger initialized successfully.")
