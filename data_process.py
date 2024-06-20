# 对数据进行一些预处理的函数
import logging
import os
import os.path as osp
import shutil
import sys
import time
from pprint import pp

import datasets
import opencc
import tokenizers
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils import get_logger

DATASET_PATH = "./dataset/full"
NEW_DATASET_PATH = "./dataset/new"
TOKENIZER_PATH = "./dataset/tokenizer.json"


os.makedirs(DATASET_PATH, exist_ok=True)

# 记录日志
logger = get_logger("DataProcess")


class DataProcess:
    def __init__(
        self,
        tokenizer=None,
        dataset_to_use="full",  # full new
        prefix='',
        reproc_full=False,
    ) -> None:
        # 提供的tokenizer最好是fast实现

        # 1. 加载处理好的全连接的数据集
        if dataset_to_use == "full":
            if len(os.listdir(DATASET_PATH)) == 0 or reproc_full:
                self.dataset = self.__preprocess_full_dataset()
            else:
                logger.info("load full dataset from disk")
                self.dataset = load_from_disk(DATASET_PATH, keep_in_memory=True)
        elif dataset_to_use == "new":
            self.dataset = load_from_disk(NEW_DATASET_PATH, keep_in_memory=True)
        else:
            raise

        # test 最终可以注释掉的代码部分
        # self.dataset = self.__preprocess_full_dataset()
        # print(self.dataset["yue"][:2])
        # print(self.dataset["zh"][:2])

        # 2. 使用连接起来的数据集训练tokenizer，(如果存在文件就读取,)
        logger.info("get tokenizer")
        if tokenizer is None:
            # 如果没有提供tokenizer 就自行训练tokenizer
            tokenizer = train_tokenizer(self.dataset, retrain=True)

            # 下面这部分主要实现动态填充，也可以给trainer传递参数
            def encode(batch):
                # tokenizer 接受str或者str列表
                # batch是数据字典，包括yue的n条数据组成的列表和zh的n条数据组成的列表
                batch = tokenizer(
                    text=batch["yue"],
                    text_target=batch["zh"],
                    padding=True,  # 最长填充
                    truncation=True,  # 截断超过最长长度的
                    max_length=128,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                # ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
                return batch

            # # https://huggingface.co/docs/datasets/process#format-transform
            self.dataset.set_transform(encode)  # 运行时调用，可以实现动态填充长度

        else:
            # 这里不实现动态填充，而是使用collator进行之后的填充对齐操作
            def encode(examples):
                # tokenizer 接受str或者str列表
                # batch是数据字典，包括yue的n条数据组成的列表和zh的n条数据组成的列表
                # prefix = "翻译粤语为简体中文:"
                batch = tokenizer(
                    text=[prefix + e for e in examples["yue"]],
                    text_target=examples["zh"],
                    truncation=True,  # 截断超过最长长度的
                    max_length=128,
                )
                # ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
                return batch

            self.dataset = self.dataset.map(encode, batched=True)

        # TEST
        # pp(self.dataset[:2])
        # pp(self.dataset[:5])
        logger.info("finish data processing")

    def get_dataset(self, test_size=0.2):
        # 分割数据集并返回
        # test_size 整数为测试集条数，小数为测试集比例
        return self.dataset.train_test_split(test_size)

    def __preprocess_full_dataset(self):
        # 1. 下载在线数据集，存储在 DATASET_CACHE，或者从中加载
        logger.info("preprocess full dataset")
        dataset1 = load_dataset(
            "botisan-ai/cantonese-mandarin-translations",
            keep_in_memory=True,
        )
        dataset2 = load_dataset(
            "raptorkwok/cantonese-traditional-chinese-parallel-corpus",
            keep_in_memory=True,
        )

        # 2. 数据格式处理为yue zh的无嵌套，无划分的数据集
        logger.info("dataset flatten, concatenate")
        rename_map = {
            "translation.yue": "yue",
            "translation.zh": "zh",
        }
        dataset1 = dataset1.flatten().rename_columns(rename_map)
        dataset2 = dataset2.flatten().rename_columns(rename_map)

        # 3. 取消原数据集的划分，在将所有数据集合并之后再创建数据集划分
        dataset1 = concatenate_datasets([dataset1["train"]])
        dataset2 = concatenate_datasets(
            [dataset2["train"], dataset2["validation"], dataset2["test"]]
        )

        # 4. 此处进行针对单个数据集的特殊处理: 比如将本来是繁体的数据转换为简体
        logger.info("special process")
        self.cc = opencc.OpenCC("t2s")

        def t2s(example):
            example["zh"] = self.cc.convert(example["zh"])
            return example

        dataset2 = dataset2.map(t2s)

        # 5. 将所有训练语料拼接起来作为一个数据集训练,并为数据集按字为单位添加空格分隔（方便tokenizer分字）
        logger.info("concat all dataset")
        full_dataset = concatenate_datasets([dataset1, dataset2])

        # 6. 数据清洗，去除空字符串
        # def filter_func(x):
        #     if len(x["yue"]) != 0 and len(x["zh"]) != 0:
        #         return True
        #     else:
        #         return False

        full_dataset = full_dataset.filter(lambda x: len(x["yue"]) != 0)

        # 保存为文件
        print("full_dataset length:", len(full_dataset))
        full_dataset.save_to_disk(DATASET_PATH)
        return full_dataset


def train_tokenizer(dataset=None, retrain=False):
    # 在自己的数据集上 训练得到tokenizer 的函数
    # 这部分属于 LSTM 方法，也可以直接用bert的tokenizer，当前的tokenizer实现不是很完善

    # 如果需要再训练或者不存在缓存的tokenizer，在这个条件下需要dataset存在
    if (retrain or not osp.exists(TOKENIZER_PATH)) and dataset is not None:
        logger.info("retrain tokenizer")
        # 创建分词器模型，使用WordLevel进行分词中文词汇
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

        # 创建 Normalizer
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.Lowercase(),
                normalizers.StripAccents(),
            ]
        )  # 转换兼容字符

        # 定义 PreTokenizer
        # 中文直接在字符意义上处理即可
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        # 之前给语料添加了空格

        # tokenizer.post_processor = processors.TemplateProcessing(
        #     single="[CLS] $0 [SEP]",
        #     # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        #     special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
        # )  # TODO 根据模型给出处理模板 比如添加特殊token

        trainer = trainers.WordLevelTrainer(
            vocab_size=10000,
            min_frequency=1,
            show_progress=True,
            # FIXME cls 是bert的分词
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            # special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                batch_text = []
                batch_text.extend(dataset[i : i + batch_size]["yue"])
                batch_text.extend(dataset[i : i + batch_size]["zh"])
                yield batch_text

        # 训练分词器

        tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))

        # 设置分词器填充 和截断
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
        tokenizer.enable_truncation(max_length=512)

        # 保存到文件
        tokenizer.save(TOKENIZER_PATH)

    # 从文件加载tokenizer
    logger.info("load tokenizer from file")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

    return tokenizer


# def fix_tokenizer():
#     # <unk>消失之谜
#     # FIXME 清洗数据中的未登录词
#     dataset = load_from_disk(DATASET_PATH)
#     tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
#     unk_dict = {}
#     for example in dataset["yue"]:
#         tokens = tokenizer.tokenize(example)
#         ids = tokenizer.convert_tokens_to_ids(tokens)
#         ids_1 = tokenizer.encode(example)

#         for i, idx in enumerate(ids):
#             if idx == tokenizer.unk_token_id:
#                 if tokens[i] not in unk_dict:
#                     unk_dict[tokens[i]] = 0
#                 unk_dict[tokens[i]] += 1

#     print(unk_dict)

#     charset = set()
#     for s in list(unk_dict.keys()):
#         charset.update(s)
#     print(charset)
#     print(len(charset))


def make_dataset(ckpt_i=0):
    # 使用api得到更好的数据集用于训练
    from interface import translate_yue_to_zh

    dataset = load_from_disk(DATASET_PATH)
    raw_path = osp.join(NEW_DATASET_PATH, "raw.txt")
    os.makedirs(NEW_DATASET_PATH, exist_ok=True)
    with open(raw_path, "a") as f:
        try:
            for i in tqdm(range(ckpt_i, len(dataset))):
                # print(dataset[i])
                if len(dataset[i]["yue"]) == 0:  # 跳过错误数据
                    print(f"skip sample{i}")
                    print(dataset[i])
                    continue
                line = [dataset[i]["yue"]]
                line.append(translate_yue_to_zh(dataset[i]["yue"]))
                f.write("\t".join(line) + "\n")
        except Exception as e:  # noqa: E722
            print(e)
            print(f"\n异常中断！中断点： {i}")
            sys.exit(1)

            # time.sleep(1)

    print(len(dataset))
    return


def trans_new_dataset():
    # 定义数据文件路径

    file_path = "./dataset/new/raw.txt"

    full_dataset = load_from_disk(DATASET_PATH)

    # 读取数据文件
    data = {"yue": [], "zh": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                yue, zh = line.strip().split("\t")
                data["yue"].append(yue)
                data["zh"].append(zh)
            except ValueError:
                yue = line.strip().split("\t")[0]
                data["yue"].append(yue)
                data["zh"].append(full_dataset[i]["zh"])
                print(i, line.strip(), full_dataset[i])

    # 将字典数据转换为 Dataset 对象
    dataset = Dataset.from_dict(data)

    # 打印一些数据以验证
    print(dataset)

    # 保存数据
    dataset.save_to_disk(NEW_DATASET_PATH)


if __name__ == "__main__":
    pass
    # 重新处理数据集（清洗
    # data = DataProcess(reproc_dataset=True)
    # dataset = data.get_dataset()
    # print(dataset)

    # pp(dataset["train"][:2])  # 展示文本数据处理后的结果

    # 清洗数据集测试
    # dataset = load_from_disk(DATASET_PATH)
    # print(len(dataset))
    # dataset = dataset.filter(lambda x: len(x["yue"]) != 0)
    # print(len(dataset))
    # dataset = dataset.filter(lambda x: len(x["zh"]) != 0)
    # print(len(dataset))

    # print(dataset[24976])
    # print(dataset[68372])
    # print(dataset[80866])
    # print(dataset[127637])

    # 创建new raw数据集 本步骤需要有至少260w 字符量的api接入，代码实现调用了科大讯飞的文本翻译api
    # make_dataset()

    # 把数据集保存为transformers dataset格式 ，顺带处理空数据
    # trans_new_dataset()

    # <unk>消失之谜 没有消失，大约2k个未登录字，但是不影响训练结果
    # dataset = load_from_disk(DATASET_PATH)
    # tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
    # print(tokenizer.unk_token)
    # ids = tokenizer.encode("圂刉媕惏")
    # tokens = tokenizer.convert_ids_to_tokens(ids)
    # fix_tokenizer()

    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(tokens)
    # print(token_ids)

    # sens = ["杞人嘅朋友嘆咗一口氣", "泥水佬開門口過得人過得自己"]
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    # # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # # tokenizer.save_pretrained("./bert_tokenizer")
    # pp(sens)
    # pp(tokenizer(sens))
    # pp(len(tokenizer))
    # # list(tokenizer.get_vocab().keys())
    # encoded = tokenizer.encode(sens[0])
    # pp(encoded)
    # pp(tokenizer.decode(encoded, skip_special_tokens=True))
    # # pp(tokenizer.token_to_id("[PAD]"))
