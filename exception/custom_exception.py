import sys
import traceback
from types import ModuleType

# Import centralized custom logger
from logger.custom_logger import CustomLogger

# Initialize logger for this module
logger = CustomLogger().get_logger(__file__)


class DocumentPortalException(Exception):
    """
    Custom exception class for the Document Portal application.

    Purpose:
    - Wraps original exceptions
    - Captures file name and line number
    - Preserves full traceback
    - Produces clean, structured logs
    """

    def __init__(self, error_message: Exception, error_details: ModuleType):
        """
        Initialize the custom exception.

        :param error_message: Original exception object
        :param error_details: sys module (used to extract traceback info)
        """

        # Initialize base Exception with readable error message
        super().__init__(str(error_message))

        # Extract exception type, value, and traceback
        _, _, exc_tb = error_details.exc_info()

        # Default values (defensive fallback)
        self.file_name = "Unknown"
        self.lineno = -1

        # If traceback exists, extract file name and line number
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno

        # Store string version of the error message
        self.error_message = str(error_message)

        # Capture full traceback as a formatted string
        self.traceback_str = "".join(
            traceback.format_exception(*error_details.exc_info())
        )

    def __str__(self) -> str:
        """
        Controls how the exception is rendered as a string.
        This representation is used when logging or printing the exception.
        """
        return (
            f"\nError in file [{self.file_name}] at line [{self.lineno}]\n"
            f"Message: {self.error_message}\n"
            f"Traceback:\n{self.traceback_str}"
        )


# Standalone test block
if __name__ == "__main__":
    try:
        # Intentionally raise an exception for testing
        a = 1 / 0
    except Exception as e:
        # Wrap original exception with custom exception
        app_exc = DocumentPortalException(e, sys)

        # Log structured exception
        logger.error(app_exc)

        # Re-raise while preserving original traceback
        raise app_exc from e
