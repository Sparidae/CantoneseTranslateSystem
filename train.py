# 训练流程
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_process import DataProcess
from metrics import compute_metrics
from model import ConfigTemplate
from utils import get_logger

logger = get_logger("FineTune_Mengzi-T5")


# 加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
print(tokenizer)


# 测试tokenizer是否能正常使用
result = tokenizer(
    ["杞 人 的 朋 友 叹 了 一 口 气", "泥 水 佬 开 门 口 过 得 人 过 得 自 己"]
)
print(result)
result = tokenizer(
    ["杞 人 的 朋 友 叹 了 一 口 气", "泥 水 佬 开 门 口 过 得 人 过 得 自 己"]
)
print(tokenizer)
