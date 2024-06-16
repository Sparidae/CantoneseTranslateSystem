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
logger.info("load t5 pretrained tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
# print(tokenizer)


# 测试tokenizer是否能正常使用
result = tokenizer(["杞人的朋友叹了一口气", "泥水佬开门口过得人过得自己"])
print(result)

# 使用tokenizer处理数据
data = DataProcess(tokenizer)

dataset = data.get_dataset()
print(dataset)
