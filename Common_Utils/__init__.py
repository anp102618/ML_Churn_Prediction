import logging
import os
import sys
import yaml
import zipfile
from pathlib import Path
from datetime import datetime
import time
import joblib
import tracemalloc
from functools import wraps

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


def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Running '{func.__name__}'...")

        start_time = time.time()
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()

        duration = end_time - start_time
        memory_used_kb = current / 1024
        peak_memory_kb = peak / 1024

        logger.info(f"'{func.__name__}' completed in {duration:.4f} sec")
        logger.info(f"Memory used: {memory_used_kb:.2f} KB (peak: {peak_memory_kb:.2f} KB)")

        return result
    return wrapper


@track_performance
def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        logging.error(f"ZIP file does not exist: {zip_path}")
        return

    if not zipfile.is_zipfile(zip_path):
        logging.error(f"Not a valid ZIP file: {zip_path}")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            logging.info(f"Extracted {len(zip_ref.namelist())} files to '{extract_to}'")

    except zipfile.BadZipFile:
        logging.exception("The ZIP file is corrupted or unreadable.")
    except PermissionError:
        logging.exception("Permission denied while extracting files.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")


def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"yaml file at path: {file_path} loaded")
        return config
    
    except CustomException as ce:
            logger.error(f"Exception found: {ce}")

def load_yaml_forTest(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"yaml file at path: {file_path} loaded")
        return config
    
    except CustomException as ce:
            print(f"Exception found: {ce}")

def load_joblib(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = joblib.load(file)
        return model
    except CustomException as e:
        logger.error(f"Error loading model from {filepath}: {e}")

def params_dict(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = joblib.load(file)
        if hasattr(model, "get_params"):
            all_params = model.get_params()
            filtered_params = {k: v for k, v in all_params.items() if v is not None and v != 0}
        return filtered_params

    except CustomException as e:
        logger.error(f"Error loading model from {filepath}: {e}")

def update_final_model_path(yaml_path, final_model_path):
    try:
        final_model_path = Path(final_model_path).as_posix()
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        config['final_model_path'] = final_model_path

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)

    except CustomException as e:
        logger.error(f"Error occured: {e}")

def append_constants(yaml_path: str, key: str, value):
    try:

        const_data = load_yaml(yaml_path)
        if key not in const_data:
            const_data[key] = value

        else :
            const_data[key] = value

        with open(yaml_path, 'w') as f:
                yaml.dump(const_data, f, default_flow_style=False)

    except CustomException as e:
        logger.error(f"Error occured: {e}")

    

def transfer_file(source_path:str, destination_path:str):

    source_file = Path(source_path)
    destination_folder = Path(destination_path)
    destination_file = destination_folder / "raw_data.csv"

    if not source_file.exists():
        logging.error(f"Source file not found: {source_file}")
    else:
        if not destination_folder.exists():
            logging.info(f"Destination folder not found. Creating: {destination_folder}")
            destination_folder.mkdir(parents=True, exist_ok=True)
        else:
            logging.info(f"Destination folder exists: {destination_folder}")

        if destination_file.exists():
            logging.info(f" File 'raw_data.csv' already exists in destination. Overwriting...")

        try:
            destination_file.write_bytes(source_file.read_bytes())
            logging.info(f" File copied successfully from: {source_file} to: {destination_file}")

        except CustomException as ce:
            logger.error(f"Exception found: {ce}")

        else:
            try:
                destination_file.write_bytes(source_file.read_bytes())
                logging.info(f" File copied successfully from: {source_file} to: {destination_file}")

            except CustomException as ce:
                logger.error(f"Exception found: {ce}")

        
        

