import sys
import traceback
from typing import Optional, cast
from types import ModuleType

# Import centralized custom logger (kept for backward compatibility)
from logger.custom_logger import CustomLogger

# Initialize logger for this module (unchanged)
logger = CustomLogger().get_logger(__file__)


class DocumentPortalException(Exception):
    """
    Custom exception class for the Document Portal application.

    Industry-grade characteristics:
    - Flexible constructor (message or exception)
    - Accurate root-cause traceback extraction
    - Preserves exception chaining
    - Logger- and API-friendly formatting
    """

    def __init__(
        self,
        error_message: object,
        error_details: Optional[object] = None,
    ):
        """
        :param error_message: Exception or custom error message
        :param error_details: sys module, Exception object, or None
        """

        # Normalize message
        if isinstance(error_message, BaseException):
            normalized_message = str(error_message)
        else:
            normalized_message = str(error_message)

        # Resolve exc_info safely
        exc_type = exc_value = exc_tb = None

        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        elif isinstance(error_details, ModuleType) and hasattr(error_details, "exc_info"):
            exc_info_obj = cast(ModuleType, error_details)
            exc_type, exc_value, exc_tb = exc_info_obj.exc_info()

        elif isinstance(error_details, BaseException):
            exc_type = type(error_details)
            exc_value = error_details
            exc_tb = error_details.__traceback__

        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the deepest traceback frame (root cause)
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # Defensive defaults
        self.file_name = "<unknown>"
        self.lineno = -1

        if last_tb and last_tb.tb_frame:
            self.file_name = last_tb.tb_frame.f_code.co_filename
            self.lineno = last_tb.tb_lineno

        self.error_message = normalized_message

        # Full traceback (optional)
        if exc_type and exc_tb:
            self.traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self) -> str:
        """
        Compact, production-safe string representation.
        Suitable for logs, APIs, and monitoring tools.
        """
        base_msg = (
            f"Error in [{self.file_name}] at line [{self.lineno}] | "
            f"Message: {self.error_message}"
        )

        if self.traceback_str:
            return f"{base_msg}\nTraceback:\n{self.traceback_str}"

        return base_msg

    def __repr__(self) -> str:
        return (
            "DocumentPortalException("
            f"file={self.file_name!r}, "
            f"line={self.lineno}, "
            f"message={self.error_message!r})"
        )


# Standalone test block (unchanged usage)
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        app_exc = DocumentPortalException(e, sys)
        logger.error(app_exc)
        raise app_exc from e
