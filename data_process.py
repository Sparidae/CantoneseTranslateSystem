# 对数据进行一些预处理的函数
from datasets import load_dataset

DATASET_CACHE = "./dataset/hf_dataset"

dataset = load_dataset(
    "botisan-ai/cantonese-mandarin-translations", cache_dir=DATASET_CACHE
)

dataset = load_dataset(
    "raptorkwok/cantonese-traditional-chinese-parallel-corpus", cache_dir=DATASET_CACHE
)
