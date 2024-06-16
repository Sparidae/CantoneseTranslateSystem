# 训练流程
from pprint import pp

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_process import DataProcess
from metrics import get_compute_metric
from model import ConfigTemplate
from utils import count_trainable_parameters, get_logger

logger = get_logger("FineTune_Mengzi-T5")
set_seed(42)

# 加载预训练的tokenizer
logger.info("load t5 pretrained tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
# tokenizer.save_pretrained("t5_tokenizer")

# 测试tokenizer是否能正常使用
# result = tokenizer(["杞人的朋友叹了一口气", "泥水佬开门口过得人过得自己"])
# print(result)

# 使用tokenizer处理数据
logger.info("process data")
data = DataProcess(tokenizer)
dataset = data.get_dataset(8192)

# FIXME 解决粤语未登录词的问题
## TEST
print(dataset)
# pp(dataset["train"][:2])

# result = tokenizer.batch_decode(dataset["train"][4:8]["input_ids"])
# pp(result)
# result = tokenizer.batch_decode(dataset["train"][4:8]["labels"])
# pp(result)


# 加载模型
logger.info("load t5 pretrained model")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Langboat/mengzi-t5-base",
    low_cpu_mem_usage=True,
)
count_trainable_parameters(model)  # 248M参数
# print(model)

# # 配置模型LoRA
# logger.info("configure LoRA finetune model")
# # 针对性微调
# # for name,parameter in model.named_parameters(): # 可以使用表达式匹配想微调的层
# #     print(name)
# config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM,
#     # modules_to_save=[], # 除了LoRA还想训练原模型的哪部分参数
# )
# model = get_peft_model(model, config)

# model.print_trainable_parameters()


# 加载评估函数
logger.info("get compute_metrics")
compute_metric = get_compute_metric(tokenizer)


# 训练参数
# 训练参数
logger.info("configure trainer")
beam_config = GenerationConfig(  # 束搜索是因为翻译评估需要稳定的输出，采样具有随机性，每次的评估都不一样
    max_new_tokens=60,  # TODO 匹配数据集的最大长度
    num_beams=3,
    early_stopping=True,
    bos_token_id=7,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
top_config = GenerationConfig()
args = Seq2SeqTrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    # num_train_epochs=3, # 默认3个
    # 梯度累计和检查点优化策略
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 进行一次更新的梯度累计步数 BS*GA=32，显示的也是这个
    gradient_checkpointing=True,
    # 评估优化
    per_device_eval_batch_size=8,
    eval_accumulation_steps=4,
    # 日志
    logging_steps=8,
    # 评估
    eval_strategy="steps",
    eval_steps=512,  # 512
    # 保存
    save_strategy="steps",
    save_steps=512,
    save_total_limit=1,  # 只保存最好的和最后的一个
    load_best_model_at_end=True,
    metric_for_best_model="bleu-4",
    # 生成
    predict_with_generate=True,
    generation_config=beam_config,
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

logger.info("start training")
trainer.train()


# from peft import PeftModel

# PeftModel.from_pretrained
