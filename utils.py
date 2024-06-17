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


def get_time_str():
    import datetime

    time_str = datetime.now().strftime("%b%d_%H-%M-%S")
    return time_str


def count_trainable_parameters(model):
    # 获取所有可训练参数的数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 将参数数量转换为百万（M）为单位
    trainable_params_m = trainable_params / 1e6
    if trainable_params_m < 1e3:
        print(f"trainable model params: {trainable_params_m:.2f}M")
    else:
        trainable_params_b = trainable_params_m / 1e3
        print(f"trainable model params: {trainable_params_b:.2f}B")
    return trainable_params


if __name__ == "__main__":
    logger = get_logger("Data")

    logger.info("msdae")
