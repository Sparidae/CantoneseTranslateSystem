# 对数据进行一些预处理的函数
import opencc
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

# from tokenizers import


DATASET_CACHE = "./dataset/hf_dataset"

# tokenizer = AutoTokenizer.from_pretrained("AlienKevin/bart-canto-mando-bing-300K-typo")


class DataProcess:
    def __init__(self) -> None:
        # 加载数据集
        dataset1 = load_dataset(
            "botisan-ai/cantonese-mandarin-translations", cache_dir=DATASET_CACHE
        )
        dataset2 = load_dataset(
            "raptorkwok/cantonese-traditional-chinese-parallel-corpus",
            cache_dir=DATASET_CACHE,
        )
        # 数据格式处理为yue zh的无嵌套，无划分的数据集
        rename_map = {
            "translation.yue": "yue",
            "translation.zh": "zh",
        }
        dataset1 = dataset1.flatten().rename_columns(rename_map)
        dataset2 = dataset2.flatten().rename_columns(rename_map)

        # 取消元数据集的划分，在拼接之后再创建数据集划分
        dataset1 = concatenate_datasets([dataset1["train"]])
        dataset2 = concatenate_datasets(
            [dataset2["train"], dataset2["validation"], dataset2["test"]]
        )

        # 数据集 特殊处理，比如将本来是繁体的数据转换为简体
        self.cc = opencc.OpenCC("t2s")

        def t2s(example):
            example["zh"] = self.cc.convert(example["zh"])
            return example

        dataset2 = dataset2.map(t2s)

        # 将所有训练语料拼接起来作为一个数据集训练
        self.dataset = concatenate_datasets([dataset1, dataset2])
        # print(len(self.dataset))
        # TODO tokenization 即时应用转换
        tokenizer = None

        def encode(batch):
            return tokenizer  # TODO

        self.dataset.set_transform(encode)

        # 将对应列的格式更改为张量格式
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        return

    def get_dataset(
        self,
    ):
        # TODO 分割数据集并返回
        return self.dataset
        pass


if __name__ == "__main__":
    data = DataProcess()
