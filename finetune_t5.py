# 训练流程
import datetime
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
from utils import count_trainable_parameters, get_logger, get_time_str

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
data = DataProcess(tokenizer, dataset_to_use="new")
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

# 配置模型LoRA
logger.info("configure LoRA finetune model")
# 针对性微调
# for name,parameter in model.named_parameters(): # 可以使用表达式匹配想微调的层
#     print(name)
config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    init_lora_weights="pissa",
    # modules_to_save=[], # 除了LoRA还想训练原模型的哪部分参数
)
model = get_peft_model(model, config)


# lora模型更新参数，大量降低微调参数量
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()


# 加载评估函数
logger.info("get compute_metrics")
compute_metric = get_compute_metric(tokenizer)


# 训练参数
# 训练参数
logger.info("configure trainer")
time_str = get_time_str()
model_name = "t5_lora_new"

beam_config = GenerationConfig(  # 束搜索是因为翻译评估需要稳定的输出，采样具有随机性，每次的评估都不一样
    max_new_tokens=128,  # TODO 匹配数据集的最大长度
    num_beams=3,
    early_stopping=True,
    bos_token_id=7,
    # no_repeat_ngram_size=2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
top_config = GenerationConfig(  # 可以作为翻译生成策略进行测试
    max_new_tokens=128,
    do_sample=True,
    top_k=20,
    top_p=0.8,
)
args = Seq2SeqTrainingArguments(
    output_dir=f"./output/{model_name}/{time_str}",
    learning_rate=2e-5,
    # num_train_epochs=3, # 默认3个
    # 优化器，调度器
    # optim="adafactor",
    # 梯度累计和检查点优化策略
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # 进行一次更新的梯度累计步数 BS*GA=32，显示的也是这个
    gradient_checkpointing=True,
    # 评估优化
    per_device_eval_batch_size=16,
    eval_accumulation_steps=1,
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


"""TODO
1. 未登录词  不做

2. 清洗数据  可做，后半部分没有的和前半部分都没有的清除掉

3. 测试lora 继续
    new数据集lora
    合并lora模型到原模型

4. 做读取模型的接口部分

5. 看LSTM进度问题 继续
"""
