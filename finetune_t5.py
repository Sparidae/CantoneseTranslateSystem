# 训练流程
from pprint import pp

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_process import DataProcess
from metrics import get_compute_metric
from model import ConfigTemplate
from utils import get_logger

logger = get_logger("FineTune_Mengzi-T5")


# 加载预训练的tokenizer
logger.info("load t5 pretrained tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")


# 测试tokenizer是否能正常使用
result = tokenizer(["杞人的朋友叹了一口气", "泥水佬开门口过得人过得自己"])
print(result)

# 使用tokenizer处理数据
data = DataProcess(tokenizer)

dataset = data.get_dataset()

# FIXME 解决粤语未登录词的问题
## TEST
print(dataset)
# pp(dataset["train"][:2])

# result = tokenizer.batch_decode(dataset["train"][4:8]["input_ids"])
# pp(result)
# result = tokenizer.batch_decode(dataset["train"][4:8]["labels"])
# pp(result)


# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("Langboat/mengzi-t5-base")
# print(model)

# collator = DataCollatorForSeq2Seq(tokenizer)

# collator()
# result = model(**dataset["train"][4:8])
# print(result)
# print(result.keys())

import evaluate
import numpy as np

bleu = evaluate.load("bleu")


def compute_metric(eval_pred):
    pred, labels = eval_pred
    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 默认为4gram bleu
    bleu_score = bleu.compute(predictions=pred, references=labels, tokenizer=tokenizer)
    return {"bleu-4": bleu_score}


# # 训练参数
args = Seq2SeqTrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    logging_steps=8,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    metric_for_best_model="bleu-4",
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metric,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
)

trainer.train()
