"""Sets up logger with different colours for logging levels."""
import logging


# Define a custom formatter with ANSI escape codes for color
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
    }

    def format(self, record) -> str:
        log_message = super(ColoredFormatter, self).format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}\033[0m"


# Function to configure a logger with colored formatting
def configure_logger(logger_name) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Define the format with fixed widths for name and levelname
    formatter = ColoredFormatter("(%(name)s) [%(levelname)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = configure_logger("gonas")


def info(message, logger=LOGGER) -> None:
    logger.info(message)


def warning(message, logger=LOGGER) -> None:
    logger.warning(message)


def error(message, logger=LOGGER) -> None:
    logger.error(message)
