import logging
import logging.config
import os
import yaml
from datetime import datetime

def configure_logger(config_path: str):
    # Create a folder for logging
    directory_path = "./log"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    # Load logger config from file
    with open(config_path, 'r') as f:
        # Load the logging configuration file
        logging_config: dict = yaml.safe_load(f.read())
        if logging_config['new_file']:
            # Create a new log file with the current time
            current_time = datetime.now().strftime('%Y%m%d_%Hh%Mmin%Ss')
            logging_config['handlers']['file_handler']['filename'] = \
                f'./log/{current_time}.log'
        else:
            # Clear the log file
            open(logging_config['handlers']['file_handler']['filename'], 'w').close()
        logging.config.dictConfig(logging_config)
    