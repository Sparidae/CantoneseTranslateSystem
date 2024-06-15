# 对数据进行一些预处理的函数
import logging
import os
import os.path as osp
from pprint import pp

import datasets
import opencc
from datasets import concatenate_datasets, load_dataset, load_from_disk
from tokenizers import Tokenizer, models, normalizers, trainers
from tokenizers.models import WordLevel, WordPiece
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils import get_logger

DATASET_PATH = "./dataset/full"
DATASET_CACHE = "./dataset/hf_dataset"
TOKENIZER_PATH = "./dataset/tokenizer.json"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(DATASET_CACHE, exist_ok=True)

# 记录日志
logger = get_logger("DataProcess")


class DataProcess:
    def __init__(self, tokenizer=None) -> None:
        # 提供的tokenizer最好是fast实现

        # 1. 加载处理好的全连接的数据集
        if len(os.listdir(DATASET_PATH)) == 0:
            self.dataset = self.__preprocess_full_dataset()
        else:
            logger.info("load full dataset from disk")
            self.dataset = load_from_disk(DATASET_PATH)

        # 2. 使用连接起来的数据集训练tokenizer，(如果存在文件就读取,)
        logger.info("get tokenizer")
        if tokenizer is None:  # 如果没有提供tokenizer 就自行训练tokenizer
            tokenizer = train_tokenizer(self.dataset, retrain=True)

        def encode(batch):
            # tokenizer 接受str或者str列表
            # batch是数据字典，包括yue的n条数据组成的列表和zh的n条数据组成的列表
            batch = tokenizer(
                text=batch["yue"],
                text_target=batch["zh"],
                padding=True,  # 最长填充
                truncation=True,  # 截断超过最长长度的
                max_length=512,
                return_tensors="pt",
            )
            # ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
            return batch

        # # https://huggingface.co/docs/datasets/process#format-transform
        self.dataset.set_transform(encode)  # 运行时调用，可以实现动态填充长度

        # 测试
        # pp(self.dataset[:2])
        # pp(self.dataset[:5])

        # 3. 将对应列的格式更改为张量格式
        # self.dataset.set_format(
        #     type="torch",
        #     columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        # )
        logger.info("finish data processing")

    def get_dataset(self, ratio=0.2):
        # 分割数据集并返回
        return self.dataset.train_test_split(ratio)

    def __preprocess_full_dataset(self):
        # 1. 下载在线数据集，存储在 DATASET_CACHE，或者从中加载
        logger.info("preprocess full dataset")
        try:
            dataset1 = load_from_disk(osp.join(DATASET_CACHE, "1"))
            dataset2 = load_from_disk(osp.join(DATASET_CACHE, "2"))
        except:  # noqa: E722
            dataset1 = load_dataset(
                "botisan-ai/cantonese-mandarin-translations",
            )
            dataset2 = load_dataset(
                "raptorkwok/cantonese-traditional-chinese-parallel-corpus",
            )
            dataset1.save_to_disk(osp.join(DATASET_CACHE, "1"))
            dataset2.save_to_disk(osp.join(DATASET_CACHE, "2"))

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
        logger.info("concat all dataset and add space")
        full_dataset = concatenate_datasets([dataset1, dataset2])

        def add_space(example):
            example["yue"] = " ".join(example["yue"])
            example["zh"] = " ".join(example["zh"])
            return example

        full_dataset = full_dataset.map(add_space)
        full_dataset.save_to_disk(DATASET_PATH)
        return full_dataset


def train_tokenizer(dataset, retrain=False):
    # 训练得到tokenizer 的函数
    # 不用tokenizer
    if retrain or not osp.exists(TOKENIZER_PATH):
        logger.info("retrain tokenizer")
        # 创建分词器模型，使用WordLevel进行分词中文词汇
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

        # 创建 Normalizer
        tokenizer.normalizer = normalizers.NFKC()  # 转换兼容字符

        # 定义 PreTokenizer
        # 中文直接在字符意义上处理即可
        tokenizer.pre_tokenizer = WhitespaceSplit()  # 之前给语料添加了空格
        # tokenizer.post_processor = None # TODO 根据模型给出处理模板 比如添加特殊tokne

        # 定义训练器
        trainer = trainers.WordLevelTrainer(
            vocab_size=30000,
            min_frequency=1,
            show_progress=True,
            # TODO 根据模型需要改动合适的特殊token
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
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


if __name__ == "__main__":
    data = DataProcess()
    dataset = data.get_dataset()
    print(dataset)

    # sens =['','']

    # tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    # pp(tokenizer.token_to_id("[PAD]"))
