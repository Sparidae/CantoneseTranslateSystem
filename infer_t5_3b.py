# 训练流程
import os
import sys
from pprint import pp

import torch
from datasets import Dataset

# 加载lora模型
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from finetune_t5_3b import PREFIX
from utils import count_trainable_parameters, get_logger

logger = get_logger("Madlad400 3b T5 inference")
# https://huggingface.co/blog/how-to-generate


class InferT53b:
    def __init__(
        self,
        lora_ckpt,  # 提供checkpoint的路径或者id ，并和提供的mode匹配
    ) -> None:
        # lora
        logger.info("load model and tokenizer")
        t5_ckpt = "jbochi/madlad400-3b-mt"
        self.tokenizer = T5Tokenizer.from_pretrained(lora_ckpt)
        o_model = T5ForConditionalGeneration.from_pretrained(
            t5_ckpt,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(o_model, model_id=lora_ckpt)
        self.model.bfloat16()
        self.model.cuda()  # 模型转移到gpu设备

        # 生成设置 搜索和采样
        self.search_config = GenerationConfig(  # 束搜索是因为翻译评估需要稳定的输出，采样具有随机性，每次的评估都不一样
            max_new_tokens=128,  # TODO 匹配数据集的最大长度
            num_beams=3,
            early_stopping=True,
            # no_repeat_ngram_size=2,
            # bos_token_id=7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.sample_config = GenerationConfig(  # 可以作为翻译生成策略进行测试
            max_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            top_k=20,
            top_p=0.7,
            early_stopping=True,
            no_repeat_ngram_size=2,
            # bos_token_id=7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def translate_yue_to_zh(
        self,
        text,
        do_sample=False,
    ):
        # 调用微调模型进行翻译
        logger.info("translate")
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
    infer_t5 = InferT53b("output/t5_madlad400_3b_new/Jun18_23-16-17/checkpoint-11264")

    while True:
        text = input("请输入粤语文本:")
        trans_text = infer_t5.translate_yue_to_zh(text, do_sample=False)
        print("翻译文本：", trans_text)
        print("-" * 50)


"""
今朝早，我喺街市買咗啲新鮮嘅水果，同埋一隻好靚嘅魚。
我最鍾意嘅粵語歌係《海闊天空》，每次聽都好有感覺。
阿媽煲嘅湯真係好味，每次飲都覺得好溫暖。
今日放工之後，我約咗朋友去食火鍋，真係期待！
我喺公園散步，見到好多小朋友喺度玩，好開心。
廣州塔嘅夜景真係好靚，每次去都覺得好震撼。
明日放假，我打算去深圳行街，聽講嗰度有好多新嘢睇。
我最鍾意嘅小食係雞蛋仔，每次都要加多啲朱古力醬。
每逢週末，我都會去書店睇書，度過一個悠閒嘅下午。
最近學緊煮飯，發現原來煮飯都幾有趣。
"""
