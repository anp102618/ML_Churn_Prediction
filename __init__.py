import logging
import os
import sys
from datetime import datetime

# 🔹 Generate Unique Log File Name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 🔹 Create Log Directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 🔹 Define Full Log File Path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 🔹 Setup Logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# 🔹 Function to Generate Detailed Error Messages
def error_message_detail(error, error_detail: sys):
    """Extracts detailed error information including file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error occurred in script: [{file_name}] at line [{exc_tb.tb_lineno}] - Message: [{str(error)}]"

# 🔹 Custom Exception Class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

# 🔹 Global Exception Handler
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Logs any unhandled exceptions globally."""
    logger.critical("Unhandled Exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler  # Register Global Exception Handler

# 🔹 Export Logger & Exceptions for Other Modules
__all__ = ["logger", "CustomException"]
