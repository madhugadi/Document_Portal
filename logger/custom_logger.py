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

    Key principles:
    - JSON logs to stdout (K8s / CloudWatch compatible)
    - Optional file logging for local debugging
    - Logger & handlers configured ONCE per process
    """

    _configured = False
    _run_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.log_file_path = os.path.join(
            self.logs_dir, f"{CustomLogger._run_id}.log"
        )

    def get_logger(self, name=__file__):
        """
        Returns a structlog logger for the given module.

        :param name: Usually __file__ of the calling module
        :return: Structlog logger instance
        """

        logger_name = os.path.basename(name)

        # -------------------------
        # Configure logging ONCE
        # -------------------------
        if not CustomLogger._configured:
            # File handler (local debugging)
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            # Console handler (stdout for K8s / Docker)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)

            # 🔥 CRITICAL: remove any existing handlers
            root_logger.handlers.clear()

            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

            # Prevent log propagation creating duplicate / empty files
            root_logger.propagate = False

            CustomLogger._configured = True

        # -------------------------
        # Structlog configuration
        # -------------------------
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(
                    fmt="iso", utc=True, key="timestamp"
                ),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)


# -------------------------
# Local test
# -------------------------
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info(
        "Logger initialized successfully",
        environment="local",
        service="document-portal",
    )
