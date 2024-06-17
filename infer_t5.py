# 训练流程
import os
import sys
from pprint import pp

import torch
from datasets import Dataset

# 加载lora模型
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_process import PREFIX
from utils import count_trainable_parameters, get_logger

logger = get_logger("T5 inference")
# https://huggingface.co/blog/how-to-generate


class InferT5:
    def __init__(self, mode="lora", ft_ckpt=None) -> None:
        # origin lora
        t5_ckpt = "Langboat/mengzi-t5-base"
        # 加载模型和tokenizer
        if mode == "origin":  # 直接微调
            ckpt = t5_ckpt
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt,
                low_cpu_mem_usage=True,
            )
        elif mode == "lora":  # lora微调
            lora_ckpt = "output/t5_lora/Jun17_14-11-08/checkpoint-9216"
            self.tokenizer = AutoTokenizer.from_pretrained(lora_ckpt)
            o_model = AutoModelForSeq2SeqLM.from_pretrained(
                t5_ckpt,
                low_cpu_mem_usage=True,
            )
            self.model = PeftModel.from_pretrained(o_model, model_id=lora_ckpt)
        else:
            raise

        self.model.cuda()  # 模型转移到gpu设备

        # 生成设置 搜索和采样
        self.search_config = GenerationConfig(  # 束搜索是因为翻译评估需要稳定的输出，采样具有随机性，每次的评估都不一样
            max_new_tokens=128,  # TODO 匹配数据集的最大长度
            num_beams=3,
            early_stopping=True,
            bos_token_id=7,
            # no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.sample_config = GenerationConfig(  # 可以作为翻译生成策略进行测试
            max_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            top_k=20,
            top_p=0.8,
        )

    def translate_yue_to_zh(
        self,
        text,
        do_sample=False,
    ):
        # 只翻译一句
        model_inputs = self.tokenizer(
            text=PREFIX + text,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
        # model_inputs
        if do_sample:
            model_outputs = self.model.generate(
                generation_config=self.sample_config,
                **model_inputs,
            )
        else:
            model_outputs = self.model.generate(
                generation_config=self.search_config,
                **model_inputs,
            )

        trans_text = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )
        return trans_text


if __name__ == "__main__":
    infer_t5 = InferT5(mode="lora")

    while True:
        text = input("请输入粤语文本:")
        trans_text = infer_t5.translate_yue_to_zh(text)
        print("翻译文本：", trans_text)
        print("-" * 50)
