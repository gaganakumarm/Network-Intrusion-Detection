import sys
from types import TracebackType


class NetworkSecurityException(Exception):
    """Custom exception for the Network Intrusion Detection project."""

    def __init__(
        self,
        error_message: str,
        error_detail: tuple[type[BaseException], BaseException, TracebackType] | None = None,
    ) -> None:
        super().__init__(error_message)
        self.error_message = self._format_error(error_message, error_detail or sys.exc_info())

    @staticmethod
    def _format_error(
        error_message: str,
        error_detail: tuple[type[BaseException] | None, BaseException | None, TracebackType | None],
    ) -> str:
        _, _, exc_tb = error_detail
        if exc_tb is None:
            return error_message

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error in file [{file_name}] at line [{line_number}]: {error_message}"

    def __str__(self) -> str:
        return self.error_message
