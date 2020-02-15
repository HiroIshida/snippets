import logging
import datetime

dtime = datetime.datetime.now()
string = dtime.strftime('%Y%m%d-%H%M%S')

logger = logging.getLogger("loggingtest")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
