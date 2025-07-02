import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance
from kneed import KneeLocator
import ot
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
# import ast
#
# def compute_interclass_scores(df, max_length, payload_direction_col='udps.bi_payload_hex'):
#     n_range = list(range(1, max_length + 1))
#     m_range = list(range(50, 201, 50))  # 50, 100, 150, 200
#
#     def payload_to_matrix(payload_seq, N, M):
#         # 如果payload_seq是字符串且是列表形式，需要转成真正的list
#         if isinstance(payload_seq, str) and payload_seq.startswith('[') and payload_seq.endswith(']'):
#             try:
#                 payload_seq = ast.literal_eval(payload_seq)
#             except Exception as e:
#                 print(f"Error parsing payload_seq: {e}")
#                 # 转失败就设为空列表
#                 payload_seq = []
#
#         matrix = []
#         for pkt in payload_seq[:N]:
#             if not isinstance(pkt, str) or len(pkt) == 0:
#                 matrix.append([0] * M)
#                 continue
#             pkt = pkt[:M * 2]  # hex string，最多M个字节 = 2*M个字符
#             pkt_bytes = []
#             for i in range(0, len(pkt), 2):
#                 try:
#                     byte_val = int(pkt[i:i + 2], 16)
#                 except ValueError:
#                     byte_val = 0
#                 pkt_bytes.append(byte_val)
#             if len(pkt_bytes) < M:
#                 pkt_bytes += [0] * (M - len(pkt_bytes))
#             else:
#                 pkt_bytes = pkt_bytes[:M]
#             matrix.append(pkt_bytes)
#
#         while len(matrix) < N:
#             matrix.append([0] * M)
#         return np.array(matrix)
#
#
#     inter_scores = []
#
#     for N in tqdm(n_range, desc='N loop'):
#         scores_for_n = []
#         for M in tqdm(m_range, desc='M loop', leave=False):
#             label_avg_matrices = {}
#             for label, group in df.groupby('label'):
#                 # sampled = group.sample(
#                 #     n=20,
#                 #     random_state=42,
#                 #     replace=False
#                 # ).reset_index(drop=True)
#                 matrices = []
#                 for idx, row in group.iterrows():
#                     payload_seq = row[payload_direction_col]
#                     if isinstance(payload_seq, str):
#                         payload_seq = payload_seq.strip().split()
#                     mat = payload_to_matrix(payload_seq, N, M)
#                     matrices.append(mat)
#                 avg_matrix = np.mean(matrices, axis=0)
#                 label_avg_matrices[label] = avg_matrix
#
#             label_list = sorted(label_avg_matrices.keys(), key=int)
#             dist_list = []
#             for i in range(len(label_list)):
#                 for j in range(i + 1, len(label_list)):
#                     mat1 = label_avg_matrices[label_list[i]]
#                     mat2 = label_avg_matrices[label_list[j]]
#                     a = np.ones((N,)) / N
#                     b = np.ones((N,)) / N
#                     cost_matrix = ot.dist(mat1, mat2, metric='euclidean')
#                     wasserstein = ot.emd2(a, b, cost_matrix)
#                     dist_list.append(wasserstein)
#
#             scores_for_n.append(np.mean(dist_list))
#         inter_scores.append(scores_for_n)
#
#     return n_range, m_range, inter_scores
#
#
#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
#
# def normalize(arr):
#     arr = np.array(arr).reshape(-1, 1)
#     return MinMaxScaler().fit_transform(arr).flatten()
#
# def moving_average(arr, k):
#     return np.convolve(arr, np.ones(k)/k, mode='valid')
#
# def moving_average_2d(arr_2d, k):
#     arr_2d = np.array(arr_2d)
#     n_rows, n_cols = arr_2d.shape
#     smoothed = []
#     for col in range(n_cols):
#         smoothed_col = np.convolve(arr_2d[:, col], np.ones(k) / k, mode='valid')
#         smoothed.append(smoothed_col)
#     # 返回的 shape 是 (n_rows - k + 1, n_cols)，转置一下更符合习惯
#     return np.array(smoothed).T
#
#
# def auto_find_best_N_smart_combined(
#     df,
#     MAX_N,
#     output_dir="annotations_smart",
#     direction_col="udps.bi_payload_hex",
#     alpha=1,
#     window=5,
#     base_best_N_single=None  # ✅ 新增参数：已有的方向/大小最佳 N
# ):
#     os.makedirs(output_dir, exist_ok=True)
#
#     print("📊 开始计算各字段评分：")
#     print(f"  🔹 处理字段：{direction_col}")
#     N_range, M_ranges, scores = compute_interclass_scores(df, MAX_N, direction_col)
#
#     scores_array = np.array(scores)
#     smoothed_scores = moving_average_2d(scores_array, window)
#
#     smooth_N_range = N_range[window - 1:]
#
#     smooth_N_range = np.array(smooth_N_range)  # 添加这一行
#
#     # ✅【新增】根据已有 best_N_single 定位截断位置
#     if base_best_N_single is not None:
#         # 平滑后的范围中找到大于等于 base_best_N_single 的 index
#         start_idx = np.where(smooth_N_range >= base_best_N_single)[0]
#         if len(start_idx) == 0:
#             print(f"⚠️ Warning: base_best_N_single={base_best_N_single} 超出平滑后的 N 范围！将使用完整曲线。")
#             start_idx = 0
#         else:
#             start_idx = start_idx[0]
#     else:
#         start_idx = 0  # 默认从头开始
#
#     smooth_N_range_sub = smooth_N_range[start_idx:]
#     smoothed_scores_sub = smoothed_scores[start_idx:, :]  # ⬅️ 所有后续操作基于截断的平滑区域
#
#     # 保存分数
#     raw_scores_dict = {direction_col: scores}
#     smoothed_scores_dict = {direction_col: smoothed_scores}
#
#     # 计算每列的 noise_threshold（用于检测停止增长点）
#     noise_thresholds = []
#     for m_idx in range(smoothed_scores_sub.shape[1]):
#         curve = smoothed_scores_sub[:, m_idx]
#         diff = np.abs(np.diff(curve[:max(3, len(curve) // 10)]))
#         noise_thresholds.append(np.percentile(diff, 90))
#
#     # 对每个 M，找到其“最合理的” N（不再显著增长）
#     best_N_idx_list = []
#     for m_idx, noise_threshold in enumerate(noise_thresholds):
#         curve = smoothed_scores_sub[:, m_idx]
#         max_idx = np.argmax(curve)
#         for i in range(max_idx):
#             if curve[max_idx] - curve[i] >= noise_threshold:
#                 continue
#             best_idx = i
#             break
#         else:
#             best_idx = max_idx
#         best_N_idx_list.append(best_idx)
#
#     # 选择具有最大得分的组合
#     best_scores = [smoothed_scores_sub[n, m] for m, n in enumerate(best_N_idx_list)]
#     best_comb_idx = np.argmax(best_scores)
#     best_N_idx = best_N_idx_list[best_comb_idx]
#     best_M_idx = best_comb_idx
#
#     # ✅还原为原始全曲线中的索引
#     best_N_idx_global = best_N_idx + start_idx
#     best_N = smooth_N_range[best_N_idx_global]
#     best_M = M_ranges[best_M_idx]
#     best_score = smoothed_scores[best_N_idx_global, best_M_idx]
#
#     print(f"\n✅ Best N = {best_N}, Best M = {best_M}, Score = {best_score:.4f}")
#
#     # ✅ 画图仍然用完整曲线
#     plt.figure(figsize=(10, 6))
#     for m_idx, M in enumerate(M_ranges):
#         score_per_n = [row[m_idx] for row in smoothed_scores_dict[direction_col]]
#         plt.plot(smooth_N_range, score_per_n, label=f'M={M}')
#
#     # 添加最佳点标记
#     plt.axvline(x=best_N, color='black', linestyle=':', label=f"Best N = {best_N}")
#     plt.scatter(
#         best_N,
#         smoothed_scores[best_N_idx_global, best_M_idx],
#         color='red',
#         zorder=5,
#         label=f"Best M = {best_M}"
#     )
#
#     plt.xlabel('N (Number of Packets)')
#     plt.ylabel('Smoothed Interclass Score')
#     plt.title('Smoothed Interclass Score vs N for Different M')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"interclass_{direction_col}_smart.png"))
#     plt.show()
#
#
#
#
# df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\04_Interpretability\TIG\TIG_13\zcx_5_class_hex_with_node_features.csv")
#
#
#
# MAX_N=100
# # 2. 计算 best_N（根据 MAX_N 再进一步自动选择最佳 N）
# # auto_find_best_N_smart_combined(df, MAX_N)
#
# # 举例你之前方向或大小分析得出最佳 N 是 30
# auto_find_best_N_smart_combined(df, MAX_N=100, direction_col="udps.bi_payload_hex", base_best_N_single=5)

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import ot  # POT: Python Optimal Transport
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update({
    'font.size': 15,                      # 所有字体大小
    'font.family': 'Times New Roman',    # 字体族（可选）
    'axes.titlesize': 15,                # 标题字体大小
    'axes.labelsize': 15,                # x/y 轴标签大小
    'xtick.labelsize': 15,               # x 轴刻度字体大小
    'ytick.labelsize': 15,               # y 轴刻度字体大小
    'legend.fontsize': 15,               # 图例字体大小
})

def payload_to_matrix(payload_seq, N, M):
    if isinstance(payload_seq, str) and payload_seq.startswith('[') and payload_seq.endswith(']'):
        try:
            payload_seq = ast.literal_eval(payload_seq)
        except Exception as e:
            print(f"Error parsing payload_seq: {e}")
            payload_seq = []

    matrix = []
    for pkt in payload_seq[:N]:
        if not isinstance(pkt, str) or len(pkt) == 0:
            matrix.append([0] * M)
            continue
        pkt = pkt[:M * 2]
        pkt_bytes = []
        for i in range(0, len(pkt), 2):
            try:
                byte_val = int(pkt[i:i + 2], 16)
            except ValueError:
                byte_val = 0
            pkt_bytes.append(byte_val)
        if len(pkt_bytes) < M:
            pkt_bytes += [0] * (M - len(pkt_bytes))
        else:
            pkt_bytes = pkt_bytes[:M]
        matrix.append(pkt_bytes)

    while len(matrix) < N:
        matrix.append([0] * M)
    return np.array(matrix)


def compute_interclass_scores_fixed_N(df, fixed_N, m_range, payload_direction_col='udps.bi_payload_hex'):
    inter_scores = []
    for M in tqdm(m_range, desc='Searching M'):
        label_avg_matrices = {}
        for label, group in df.groupby('label'):
            sampled = group.sample(n=len(group), random_state=42, replace=False).reset_index(drop=True)
            matrices = []
            for _, row in sampled.iterrows():
                payload_seq = row[payload_direction_col]
                mat = payload_to_matrix(payload_seq, fixed_N, M)
                matrices.append(mat)
            avg_matrix = np.mean(matrices, axis=0)
            label_avg_matrices[label] = avg_matrix

        label_list = sorted(label_avg_matrices.keys(), key=int)
        dist_list = []
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                mat1 = label_avg_matrices[label_list[i]]
                mat2 = label_avg_matrices[label_list[j]]
                a = np.ones((fixed_N,)) / fixed_N
                b = np.ones((fixed_N,)) / fixed_N
                cost_matrix = ot.dist(mat1, mat2, metric='euclidean')
                wasserstein = ot.emd2(a, b, cost_matrix)
                dist_list.append(wasserstein)
        inter_scores.append(np.mean(dist_list))
    return inter_scores



def normalize(arr):
    arr = np.array(arr).reshape(-1, 1)
    return MinMaxScaler().fit_transform(arr).flatten()
def auto_find_best_M_fixed_N(
    df,
    MAX_N,
    output_dir="truncation_results",
    direction_col="udps.bi_payload_hex",
    m_range=list(range(50, 201, 1)),
    smooth_window=3
):
    os.makedirs(output_dir, exist_ok=True)

    print("✅第四步：开始寻找best_N，固定 N =", MAX_N)
    inter_scores = compute_interclass_scores_fixed_N(df, MAX_N, m_range, direction_col)

    # 平滑处理
    # smoothed_scores = uniform_filter1d(inter_scores, size=smooth_window, mode='nearest')
    # smoothed_scores=inter_scores

    inter_scores = normalize(inter_scores)

    # best_idx = np.argmax(smoothed_scores)
    # best_M = m_range[best_idx]
    # best_score = smoothed_scores[best_idx]

    #单个字节性价比
    # score_per_byte = [s / m for s, m in zip(smoothed_scores, m_range)]
    # best_idx_ratio = np.argmax(score_per_byte)
    # best_M_ratio = m_range[best_idx_ratio]
    # best_score_ratio = smoothed_scores[best_idx_ratio]


    # 信息利用率稳定点（平稳性原则）
    # 原则：找第一个使得 score 增长趋势 变得平稳/趋近饱和 的点，而不是最大值。
    diffs = np.diff(inter_scores)
    mean_diff = np.mean(diffs)
    best_idx = np.argmax(diffs < mean_diff)
    best_M = m_range[best_idx]
    best_score = inter_scores[best_idx]


    # print(f"\n✅ Fixed N = {MAX_N}, Best M = {best_M}, Score = {best_score:.4f}")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(m_range, inter_scores, label="Payload",linestyle = '--', color = 'blue')  # ← 去掉 marker='o'
    plt.axvline(x=best_M, color='black', linestyle=':', label=f"M_best = {best_M}")

    plt.axhline(y=best_score, color='black', linestyle=':',
                label=f"InterclassGap_M_best = {best_score:.3f}")
    plt.scatter(best_M, best_score, color='red', zorder=5)
    plt.xlabel("M (Bytes per Packet)")
    plt.ylabel("InterClassGap (Normalized)")
    # plt.title(f"Interclass Score vs M (Fixed N = {MAX_N})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"M_best.png"))
    plt.show()

    return best_M, best_score

# df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\04_Interpretability\TIG\TIG_13\vnat_9_class_hex_with_node_features(without 10).csv")
#
# best_M, best_score = auto_find_best_M_fixed_N(df, MAX_N=41, direction_col="udps.bi_payload_hex")
