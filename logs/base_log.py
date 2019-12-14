from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG

def create_logger(EXP_VERSION):
    log_file = ("logs/{}.log".format(EXP_VERSION)).resolve()

    logger = getLogger(EXP_VERSION, mode="w")
    logger.setLevel(DEBUG)

    formatter = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_logger(EXP_VERSION):
    return getLogger(EXP_VERSION)