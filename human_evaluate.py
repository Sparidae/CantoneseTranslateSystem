import json
import os
import random
import shutil
import time
from datetime import datetime
from pprint import pp

from infer_t5 import InferT5
from infer_t5_3b import InferT53b
from interface import UPLOAD_DIR, get_speech2text_api, get_translate_api
from utils import get_logger, get_time_str

cache = "./output/human_eval.json"

# 数据
yue_text = [
    "今朝早，我喺街市買咗啲新鮮嘅水果，同埋一隻好靚嘅魚。",
    "阿媽煲嘅湯真係好味，每次飲都覺得好溫暖。",
    "你食咗饭未？",  #
    "今日个天好靓，不如我哋一齐去海边散步，顺便食啲海鲜。",
    "你听讲过最近有一部好好睇嘅电影上映咗吗？我哋可以一齐去睇。",  #
    "你记唔记得我哋上次去旅行嗰阵，嗰间酒店真係好靓。",
    "收工唔落閘，唔通打開對門俾你入嚟坐呀",
    "黃師傅去長洲贊端路幫個後生仔睇跌打",  #
    "如果未去過葵涌嘅健康街就真係要搵個時間去行吓",
    "我哋喺度食饭嗰阵，突然落咗一场好大嘅雨。",
    "何老師問現正在錦田錦上路勸學生早啲返屋企",
]

results = {}


def generate(re_process=False):
    results = {}
    if re_process or not os.path.exists(cache):
        # # t5 full

        model = InferT5(
            "output/t5_lora_full/Jun17_14-11-08/checkpoint-9216", mode="lora"
        )

        # model.translate_yue_to_zh()
        text = model.translate_yue_to_zh(yue_text.copy())
        results["t5_full"] = text
        del model

        # # t5 new

        model = InferT5(
            "output/t5_lora_new/Jun17_17-54-40/checkpoint-13312", mode="lora"
        )

        # model.translate_yue_to_zh()
        text = model.translate_yue_to_zh(yue_text.copy())
        results["t5_new"] = text
        del model

        # # t53b full

        model = InferT53b("output/t5_madlad400_3b_full/Jun20_00-11-14/checkpoint-11776")

        # model.translate_yue_to_zh()
        text = model.translate_yue_to_zh(yue_text.copy())
        results["t53b_full"] = text
        del model

        # # t53b new

        model = InferT53b("output/t5_madlad400_3b_new/Jun18_23-16-17/checkpoint-11264")

        # model.translate_yue_to_zh()
        text = model.translate_yue_to_zh(yue_text.copy())
        results["t53b_new"] = text
        del model

        # # api

        translate_api = get_translate_api()

        text = []
        for t in yue_text.copy():
            text.append(translate_api.translate_yue_to_zh(t))

        results["api"] = text

        with open(cache, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print("数据处理完毕")
        exit(0)
    else:
        with open(cache, "r") as json_file:
            results = json.load(json_file)

    return results


def read_result(path):
    with open(path, "r") as json_file:
        results = json.load(json_file)

    return results


if __name__ == "__main__":
    results = generate()
    # read_result('./human_eval.json')

    model_scores = {m: 0 for m in results.keys()}

    for i in range(len(yue_text)):
        # 对每个文本评价好坏
        print("=" * 60)
        os.system("clear")

        print("请输入数字id 0-4 不需要分割")
        origins = yue_text[i]
        candidates = []
        for m in results.keys():
            candidates.append((m, results[m][i]))

        print("原文：", origins)
        random.shuffle(candidates)
        print("候选：")
        for i, c in enumerate(candidates):
            print(f"{i},{c[1]}")  # model,text 取出model
            print()

        print("-" * 30)

        # 打分
        good = input("好 的句子对应的数字id:")
        print()
        bad = input("差 的句子对应的数字id:")

        good_ids = [int(s) for s in good]
        bad_ids = [int(s) for s in bad]
        # 统计
        for idx in good_ids:
            model_scores[candidates[idx][0]] += 1

        for idx in bad_ids:
            model_scores[candidates[idx][0]] -= 1

        for m in model_scores.keys():
            model_scores[m] += 1
        # print(model_scores)

        # os.system("pause")

    model_scores = {m: s / len(yue_text) for m, s in model_scores.items()}

    pp(model_scores)
