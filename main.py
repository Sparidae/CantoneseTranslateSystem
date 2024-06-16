import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AdamW,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)

from data_process import DataProcess
from metrics import compute_metrics
from model import ConfigTemplate
from utils import get_logger

logger = get_logger("Main")


def parse_args():
    # 解析命令行参数 需要设置
    time_str = datetime.now().strftime("%m%d-%H%M")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expname",
        type=str,
        default=f"{time_str}",
        help="",
    )
    parser.add_argument("--hidden_size", type=int, default=512, help="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 加载命令行配置
    args = parse_args()

    # 加载数据
    logger.info("Load Data")
    data = DataProcess()
    dataset = data.get_dataset(test_size=0.2)
    print(dataset)
    print(dataset["train"][1])

    # 创建模型配置
    # config = ModelConfig()

    # 创建模型
    # model = Model()

    # 加载训练参数
    # training_arg = TrainingArguments(
    #     output_dir=f"./checkpoints/__",
    #     eval_strategy="steps",
    #     num_train_epochs=10,
    #     per_device_train_batch_size=64,
    #     per_device_eval_batch_size=64,
    #     learning_rate=3e-4,
    #     warmup_ratio=0.05,
    #     logging_dir=f"./logs/__",
    #     logging_strategy="steps",
    #     logging_steps=100,
    #     # eval_steps=100, # 默认和 logging_step相同
    #     save_strategy="steps",  # 检查点保存策略
    #     save_steps=100,  # 多少步保存检查点
    #     save_total_limit=2,  # 最多保存几个检查点
    #     load_best_model_at_end=True,
    #     metric_for_best_model="loss",
    #     # greater_is_better=False, # 默认
    # )

    # 加载trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_arg,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     optimizers=(optimizer, None) if args.mode == "finetune" else (None, None),
    #     compute_metrics=None if config.is_pretrain else compute_metrics,
    #     callbacks=[EarlyStoppingCallback(3)],
    # )

    # 训练
    # trainer.train()
    # print("=" * 80)
    # trainer.evaluate()
