# 实现评估指标
import evaluate
import numpy as np

from utils import get_logger

logger = get_logger("metrics")

# 加载 BLEU 指标
logger.info("load metrics")

bleu = evaluate.load("bleu")


def get_compute_metric(tokenizer):
    def compute_metrics(eval_pred):
        pred, labels = eval_pred
        pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 默认为4gram bleu
        # 使用统一的空格分词评价，使bleu分数具有可比性
        # bleu_2 = bleu.compute(
        #     predictions=[" ".join(p) for p in pred],
        #     references=[" ".join(p) for p in labels],
        #     max_order=4,
        # )
        bleu_4 = bleu.compute(
            predictions=[" ".join(p) for p in pred],
            references=[" ".join(p) for p in labels],
            max_order=4,
        )
        return {"bleu-4": bleu_4["bleu"]}

    return compute_metrics


#####  下面全部是测试部分  #####

# metrics_list = evaluate.list_evaluation_modules()
# print("指标的总数: ", len(metrics_list))
# print("所有的指标: ", metrics_list)
# for m in metrics_list:
#     if "bleu" in m:
#         print(m)


# TEST
# predictions = ["this is a test", "another test"]  # 预测列表 str
# references = [
#     ["this is a test"],
#     ["another one test"],
# ]  # 参考列表，每条预测可以对应多条参考,也可以只对应一条

# # 说明文档 https://huggingface.co/spaces/evaluate-metric/bleu
# # 计算 BLEU 分数
# results = bleu.compute(predictions=predictions, references=references)

# # 添加 tokenizer 来进行粤语分词
# # results = bleu.compute(predictions=predictions, references=references,tokenizer = '')

# print(f"BLEU Score: {results['bleu']}")
# print(f"BLEU Score: {results}")
