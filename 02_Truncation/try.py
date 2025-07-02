



input_str = """
1043
334
100
205
5254
"""
lines = input_str.strip().split('\n')

result = " & ".join([n for n in lines])
print(result)




def process_numbers(input_str):
    lines = input_str.strip().split('\n')
    processed = []
    for val in lines:
        try:
            f = float(val)
            if f == 0:
                processed.append('-')
            else:
                processed.append(f"{f:.3f}")
        except:
            processed.append('-')
    return processed

input_str = """
0.997093023
0.9915966386554622
0.9917695473251029
0.9998722697662537
"""

nums = process_numbers(input_str)
result = " & ".join([n if n == '-' else f"{float(n):.3f}" for n in nums])
print(result)

#
#
#
# metrics = [0.4444444444444444, 0.6666666666666666, 0.7517241379310344, 0.8828451882845189,
#            0.6363636363636364, 0.32786885245901637, 0.5429447852760736, 0.46905537459283386,
#            0.9438202247191011, 0.4594594594594595, 0.943089431, 0.6696428571428571,
#            0.8333333333333334, 0.5945945945945946]
#
# supports = [25, 23, 136, 236, 26, 83, 229, 277, 92, 37, 166, 123, 44, 63]
#
# # 拼接结果
# formatted = " & ".join([f"{m:.3f} (n={s})" for m, s in zip(metrics, supports)])
#
#
#
# print(formatted)

def process_numbers(input_str):
    lines = input_str.strip().split('\n')
    processed=0
    m=0
    for val in lines:
        f = val
        if f !='-':
            f = float(val)
            processed+=f
            m+=1
    return processed,m
#
# input_str = """
# -
# -
# -
# 0.5813953488372093
# -
# -
# 0.5077720207253886
# 0.7794486215538847
# -
# -
# -
# 0.588957055
# -
# 0.9868766404199475
# """




# 打印每类 accuracy
# for idx, acc in enumerate(accuracies):
#     print(f"Class {idx}: Accuracy = {acc:.4f}")
#
# macro_accuracy = sum(accuracies) / len(accuracies)
# print(f"\nMacro Accuracy = {macro_accuracy:.4f}")


# import numpy as np
# from sklearn.metrics import classification_report, precision_score, recall_score
#
#
# text = '''
# 344 0 0 0
# 0 311 48 1
# 0 203 38 0
# 7 0 0 7821
# '''
#
# lines = text.strip().split('\n')
# matrix = [list(map(int, line.split())) for line in lines]
# conf_matrix = np.array(matrix)
#
# # 生成真实标签和预测标签
# true = []
# pred = []
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         true.extend([i] * conf_matrix[i, j])
#         pred.extend([j] * conf_matrix[i, j])
#
# # 打印分类报告
# print(classification_report(true, pred, digits=5))
#
# # 计算 Macro Precision 和 Recall
# macro_precision = precision_score(true, pred, average='macro')
# macro_recall = recall_score(true, pred, average='macro')
# print(f"Macro Precision: {macro_precision:.5f}")
# print(f"Macro Recall:    {macro_recall:.5f}")
#
# # 每类 accuracy 计算
# num_classes = conf_matrix.shape[0]
# class_accuracies = []
#
# for i in range(num_classes):
#     TP = conf_matrix[i, i]
#     FP = conf_matrix[:, i].sum() - TP
#     FN = conf_matrix[i, :].sum() - TP
#     TN = conf_matrix.sum() - (TP + FP + FN)
#     acc = (TP + TN) / conf_matrix.sum()
#     class_accuracies.append(acc)
#     print(f"Class {i} accuracy: {acc:.3f}")
#
# # macro accuracy
# macro_accuracy = sum(class_accuracies) / num_classes
# print(f"Macro Accuracy: {macro_accuracy:.3f}")


import numpy as np

# 混淆矩阵文本（类数为4）
text = '''
0 0 8 3 0 0 4 7 0 0 3 0 0 0
0 0 7 5 0 0 2 5 1 0 3 0 0 0
0 0 52 20 0 1 12 19 7 0 25 0 0 0
0 0 12 166 0 0 5 19 6 0 28 0 0 0
0 0 2 5 0 1 5 7 0 0 6 0 0 0
0 0 3 10 0 3 8 32 0 0 27 0 0 0
0 0 4 127 0 2 33 51 1 0 11 0 0 0
0 0 4 70 0 3 22 122 2 0 53 1 0 0
0 0 0 57 0 0 0 4 20 0 11 0 0 0
0 0 3 7 0 0 4 15 1 0 7 0 0 0
0 0 1 15 0 2 4 29 1 0 113 0 0 1
0 0 6 71 0 1 5 22 2 0 16 0 0 0
0 0 1 12 0 1 3 11 1 0 15 0 0 0
0 0 17 8 0 0 1 21 2 0 14 0 0 0
'''

# 构建混淆矩阵
lines = text.strip().split('\n')
matrix = [list(map(int, line.split())) for line in lines]
conf_matrix = np.array(matrix)

num_classes = conf_matrix.shape[0]
total = conf_matrix.sum()

# 输出每类的 TP, FP, FN, TN 和 FPR
for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    TN = total - (TP + FP + FN)

    # 计算 FPR 并解释
    if (FP + TN) == 0:
        fpr = None
        explanation = "FPR 无定义（FP + TN = 0）"
    else:
        fpr = FP / (FP + TN)
        if fpr == 0:
            if FP == 0:
                explanation = "FPR 为 0（无误报，合理）"
            else:
                explanation = "FPR 为 0（但 FP ≠ 0，异常）"
        else:
            explanation = "FPR > 0（有误报）"

    # 打印结果
    if fpr == 0:
        if FP != 0:
            print(f"Class {i}:")
            print(f"  TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
            print(f"  FPR = {fpr:.3f}" if fpr is not None else "  FPR = 无定义")
            print(f"  解释：{explanation}")
            print("-" * 40)
