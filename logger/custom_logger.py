import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    """
    Custom structured logger designed for:
    - Local development (VS Code)
    - Docker containers
    - Kubernetes / AWS EKS

    Key idea:
    - Logs are emitted as JSON to stdout (required by Kubernetes)
    - Optional file logging for local debugging
    """

    def __init__(self, log_dir="logs"):
        # Create a logs directory (useful locally; ignored by EKS)
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create a timestamp-based log file name
        # Each run gets a new file
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        """
        Returns a structlog logger for the given module.

        :param name: Usually __file__ of the calling module
        :return: Structlog logger instance
        """

        # Use file name as logger name (helps identify source module)
        logger_name = os.path.basename(name)

        # -------------------------
        # Standard logging handlers
        # -------------------------

        # File handler → writes raw JSON logs to file (local debugging)
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)

        # IMPORTANT:
        # %(message)s ensures we print ONLY the JSON produced by structlog
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Console handler → logs to stdout
        # Kubernetes / EKS ONLY reads stdout/stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure Python's standard logging ONCE
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # structlog renders JSON
            handlers=[console_handler, file_handler],
        )

        # -------------------------
        # Structlog configuration
        # -------------------------

        structlog.configure(
            processors=[
                # Add ISO-8601 timestamp in UTC
                structlog.processors.TimeStamper(
                    fmt="iso", utc=True, key="timestamp"
                ),

                # Add log level field (info, error, etc.)
                structlog.processors.add_log_level,

                # Rename default "event" key for clarity
                structlog.processors.EventRenamer(to="event"),

                # Render everything as JSON (final output)
                structlog.processors.JSONRenderer(),
            ],

            # Use standard logging underneath
            logger_factory=structlog.stdlib.LoggerFactory(),

            # Cache logger for performance
            cache_logger_on_first_use=True,
        )

        # Return a structlog logger
        return structlog.get_logger(logger_name)


# Local test: run this file directly to verify logger
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("Logger initialized successfully", environment="local")
