# 训练流程
from pprint import pp

import torch
from datasets import Dataset
from peft import peft_model
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


# 测试tokenizer是否能正常使用
result = tokenizer(["杞人的朋友叹了一口气", "泥水佬开门口过得人过得自己"])
print(result)

# 使用tokenizer处理数据
data = DataProcess(tokenizer)

dataset = data.get_dataset()

# FIXME 解决粤语未登录词的问题
## TEST
# print(dataset)
# pp(dataset["train"][:2])

# result = tokenizer.batch_decode(dataset["train"][4:8]["input_ids"])
# pp(result)
# result = tokenizer.batch_decode(dataset["train"][4:8]["labels"])
# pp(result)


# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("Langboat/mengzi-t5-base")
# print(model)
result = model(dataset["train"][4:8])
print(result)

# # 训练参数
# args = Seq2SeqTrainingArguments(
#     output_dir="./output",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=8,
#     logging_steps=8,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     metric_for_best_model="rouge-l",
#     predict_with_generate=True,
# )

# # trainer
# trainer = Seq2SeqTrainer(
#     args=args,
#     model=model,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     compute_metrics=compute_metrics,
#     # tokenizer=tokenizer,
#     # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
# )

# # 训练
# trainer.train()
