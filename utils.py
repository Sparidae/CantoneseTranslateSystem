import logging
import sys


def get_logger(name, level=logging.INFO):
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = get_logger("Data")

    logger.info("msdae")
