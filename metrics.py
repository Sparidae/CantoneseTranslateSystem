# 实现评估指标
import evaluate
import numpy as np

from utils import get_logger

logger = get_logger("metrics")


def get_compute_metric(tokenizer):
    def compute_metrics(eval_pred):
        pred, labels = eval_pred
        pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 默认为4gram bleu
        bleu_score = bleu.compute(
            predictions=pred, references=labels, tokenizer=tokenizer
        )
        return {"bleu-4": bleu_score["bleu"]}

    return compute_metrics


# metrics_list = evaluate.list_evaluation_modules()
# print("指标的总数: ", len(metrics_list))
# print("所有的指标: ", metrics_list)
# for m in metrics_list:
#     if "bleu" in m:
#         print(m)


# 加载 BLEU 指标
logger.info("load metrics")
bleu = evaluate.load("bleu")


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
