import logging
import os
from datetime import datetime

LOG_FILE= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ## name of the log file it will give the name as time at which it is executed

log_path= os.path.join(os.getcwd(),"logs") ## it will give the of current working directory(cwd) and folder name will be logs

os.makedirs(log_path,exist_ok=True)

LOG_FILEPATH= os.path.join(log_path,LOG_FILE)

logging.basicConfig(level= logging.INFO,
        filename= LOG_FILEPATH,
        format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
