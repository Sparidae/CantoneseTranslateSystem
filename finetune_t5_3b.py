# 调用来自google的预训练模型google/madlad400-3b-mt
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
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)

from data_process import DataProcess
from metrics import get_compute_metric
from utils import count_trainable_parameters, get_logger, get_time_str

logger = get_logger("FineTune_madlad400-3b-mt")
set_seed(42)

PREFIX = "<2zh> "

# 加载tokenizer
tokenizer = T5Tokenizer.from_pretrained("jbochi/madlad400-3b-mt")

# 加载数据
logger.info("process data")
data = DataProcess(tokenizer, dataset_to_use="new", prefix=PREFIX)
dataset = data.get_dataset(8192)


# LoRA配置 + 半精度模型（bf16
model = T5ForConditionalGeneration.from_pretrained(
    "jbochi/madlad400-3b-mt",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    init_lora_weights="pissa",
    # modules_to_save=[], # 除了LoRA还想训练原模型的哪部分参数
)
model = get_peft_model(model, config)
model.bfloat16()

# 配置
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# 加载评估函数
logger.info("get compute_metrics")
compute_metric = get_compute_metric(tokenizer)

# # 训练参数
logger.info("configure trainer")
time_str = get_time_str()
model_name = "t5_madlad400_3b_new"

# 配置参数
beam_config = GenerationConfig(  # 束搜索是因为翻译评估需要稳定的输出，采样具有随机性，每次的评估都不一样
    max_new_tokens=128,  # TODO 匹配数据集的最大长度
    num_beams=3,
    early_stopping=True,
    # bos_token_id=7,
    # no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
args = Seq2SeqTrainingArguments(
    output_dir=f"./output/{model_name}/{time_str}",
    learning_rate=2e-5,
    # num_train_epochs=3, # 默认3个
    # 优化器，调度器
    optim="adafactor",
    # 梯度累计和检查点优化策略
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # 进行一次更新的梯度累计步数 BS*GA=32，显示的也是这个
    gradient_checkpointing=True,
    # 评估优化
    per_device_eval_batch_size=16,
    # eval_accumulation_steps=1,
    # 日志
    logging_dir=f"./output/{model_name}/{time_str}",
    logging_steps=32,
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

logger.info(f"start training {model_name} {time_str}")
trainer.train()


# # 调用示例
# text = "<2zh> I love pizza!"
# input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
# outputs = model.generate(input_ids=input_ids)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
