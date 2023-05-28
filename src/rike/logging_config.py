import logging

# Configure the root logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a separate logger object for your scripts
logger = logging.getLogger('my_script_logger')
logger.setLevel(logging.WARNING)

# Create a file handler and set its level to DEBUG
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.WARNING)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)