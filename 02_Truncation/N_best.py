import pandas as pd
from scipy.stats import wasserstein_distance
from kneed import KneeLocator
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 15,                      # 所有字体大小
    'font.family': 'Times New Roman',    # 字体族（可选）
    'axes.titlesize': 15,                # 标题字体大小
    'axes.labelsize': 15,                # x/y 轴标签大小
    'xtick.labelsize': 15,               # x 轴刻度字体大小
    'ytick.labelsize': 15,               # y 轴刻度字体大小
    'legend.fontsize': 15,               # 图例字体大小
})

output_dir = "truncation_results"

def find_best_N_for_85_coverage(
    df,
    direction_col,
    plot=True,
    coverage_threshold=0.85,
    max_node=100  # 最大长度上限
):
    # 统计每个流的长度
    lengths = df[direction_col].dropna().apply(lambda x: len(str(x).replace(',', ' ').split()))
    lengths = sorted(lengths)

    # 枚举每个可能的max_length，统计覆盖率，限制最大不超过 max_node
    max_possible = min(lengths[-1], max_node)
    coverages = []
    for n in range(5, max_possible + 1):
        coverage = np.mean(np.array(lengths) <= n)
        coverages.append((n, coverage))

    # 找到最小的N，使得覆盖率 ≥ coverage_threshold
    suggested_n = max_possible  # 默认是上限
    for n, cov in coverages:
        if cov >= coverage_threshold:
            suggested_n = n
            break

    if plot:
        ns, covs = zip(*coverages)
        plt.plot(ns, covs, label='Coverage curve')
        plt.axvline(suggested_n, color='red', linestyle='--', label=f'Suggested N_max = {suggested_n}({np.mean(np.array(lengths) <= suggested_n) * 100:.1f}%)')
        plt.axhline(coverage_threshold, color='gray', linestyle=':')
        plt.xlabel("N")
        # plt.xlim(5, 100)  # 确保横轴从 window 开始
        plt.ylabel("Coverage Rate")
        # plt.title("Coverage vs max_length (cap = {})".format(max_length_cap))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"N_max.png"))
        plt.show()

    print(f"✅第二步：MAX_N == {suggested_n}",f"📦 MAX_N 覆盖了 {np.mean(np.array(lengths) <= suggested_n) * 100:.1f}% 的流")

    return suggested_n

def compute_interclass_scores(df, max_length,direction_col):
    inter_scores = []
    n_range = range(1, max_length + 1)

    for N in tqdm(n_range, desc='Searching N'):
        label_avg_vectors = {}
        if direction_col == 'udps.bi_flow_pkt_direction':
            for label, group in df.groupby('label'):
                sampled = group.sample(
                    n=len(group),
                    random_state=42,
                    replace=False
                ).reset_index(drop=True)

                matrix = np.zeros((len(sampled), N))
                matrix.fill(-1)

                for i in range(len(sampled)):
                    direction_str = sampled.iloc[i][direction_col]
                    direction_seq = direction_str.strip().split()[:N]
                    direction_array = [int(x) for x in direction_seq]
                    matrix[i, :len(direction_array)] = direction_array

                masked_matrix = np.ma.masked_equal(matrix, -1)
                avg_vector = masked_matrix.mean(axis=0).filled(0)
                label_avg_vectors[label] = avg_vector
                # label_avg_vectors[label] = matrix.mean(axis=0)
        elif direction_col == 'udps.bi_pkt_size':
            for label, group in df.groupby('label'):
                sampled = group.sample(n=len(group), random_state=42).reset_index(drop=True)
                matrix = np.zeros((len(sampled), N))
                matrix.fill(0)

                for i in range(len(sampled)):
                    packet_size = sampled.iloc[i]['udps.bi_pkt_size']
                    size_seq = packet_size.strip().split()[:N]
                    size_array = [int(x) for x in size_seq]
                    matrix[i, :len(size_array)] = size_array
                    # print('matrix[i]',matrix[i])
                # 保存当前类别的平均向量
                label_avg_vectors[label] = matrix.mean(axis=0)
                # masked_matrix = np.ma.masked_equal(matrix, -1)
                # avg_vector = masked_matrix.mean(axis=0).filled(0)
                # label_avg_vectors[label] = avg_vector

        # compute mean Wasserstein distance between all pairs
        label_list = sorted(label_avg_vectors.keys(), key=int)
        dist_list = []
        for i in range(len(label_list)):
            for j in range(i+1, len(label_list)):
                vec1 = label_avg_vectors[label_list[i]]
                vec2 = label_avg_vectors[label_list[j]]
                dist = wasserstein_distance(vec1, vec2)
                dist_list.append(dist)

        inter_scores.append(np.mean(dist_list))

    return list(n_range), inter_scores

def plot_knee(x, y, output_path):
    kneedle = KneeLocator(x, y, curve="convex", direction="increasing")
    knee_x = kneedle.knee
    knee_y = y[knee_x - 1] if knee_x else None

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', label='InterClassScore Curve')
    if knee_x:
        plt.axvline(x=knee_x, color='red', linestyle='--', label=f"Knee = {knee_x}")
        plt.scatter(knee_x, knee_y, color='red', zorder=5)
    # plt.title("InterClassScore vs. Direction Length N")
    # plt.xlabel("Direction Length N")
    # plt.ylabel("Average InterClass Wasserstein Distance")
    plt.xlabel("N")
    plt.ylabel("InterClassGap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    return knee_x

def plot_max(x, y, output_path):
    max_idx = int(np.argmax(y))
    max_x = x[max_idx]
    max_y = y[max_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', label='InterClassScore Curve')
    plt.axvline(x=max_x, color='green', linestyle='--', label=f"N_best = {max_x}")
    plt.scatter(max_x, max_y, color='green', zorder=5)
    # plt.title("InterClassScore vs. Direction Length N")
    plt.xlabel("N")
    plt.ylabel("InterClassGap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    return max_x

def plot_near_max(x, y, output_path, ratio=0.95):
    max_score = max(y)
    threshold = ratio * max_score

    for i, score in enumerate(y):
        if score >= threshold:
            near_max_x = x[i]
            near_max_y = score
            break
    else:
        near_max_x = x[-1]
        near_max_y = y[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', label='InterClassScore Curve')
    plt.axhline(y=max_score, color='gray', linestyle='--', label=f"Ratio == {ratio}")
    plt.axvline(x=near_max_x, color='green', linestyle='--', label=f"N_best = {near_max_x}")
    plt.scatter(near_max_x, near_max_y, color='green', zorder=5)
    # plt.title(f"InterClassScore vs. Direction Length N (≥ {ratio*100:.0f}% Max)")
    plt.xlabel("N")
    plt.ylabel("InterClassGap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    return near_max_x

# 加速度
def auto_find_best_N_no_hyperparam(df, MAX_N,output_dir="annotations_no_hyperparam"):
    os.makedirs(output_dir, exist_ok=True)
    N_range, scores = compute_interclass_scores(df,MAX_N,'udps.bi_flow_pkt_direction')

    # 计算斜率（梯度）和加速度（2阶导）
    slopes = np.gradient(scores)
    accelerations = np.gradient(slopes)

    # 寻找第一个加速度从正变负的位置（增长速度变缓）
    candidate_indices = np.where((accelerations[:-1] > 0) & (accelerations[1:] <= 0))[0]

    if len(candidate_indices) > 0:
        best_idx = candidate_indices[0] + 1
        reason = "Detected information saturation (acceleration turns negative)."
    else:
        # 如果没有找到“饱和点”，退而求其次，选最大分数点
        best_idx = int(np.argmax(scores))
        reason = "No knee detected, fallback to global max."

    best_N = N_range[best_idx]
    print(f"✔️ 自动检测最佳方向长度 N = {best_N}（{reason}）")

    # 画图（高亮显示 best_N）
    plt.figure(figsize=(10, 6))
    plt.plot(N_range, scores, marker='o', label='InterClassScore Curve')
    plt.axvline(x=best_N, color='green', linestyle='--', label=f"N_best = {best_N}")
    plt.scatter(best_N, scores[best_idx], color='green', zorder=5)
    # plt.title("InterClassScore vs. Direction Length N (Auto no-hyperparam)")
    plt.xlabel("N")
    plt.ylabel("InterClassGap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interclass_score_auto_no_hyperparam.png"))
    plt.show()

    return best_N, N_range, scores

# 主函数
def auto_find_best_N(df, MAX_N,direction_cols="udps.bi_flow_pkt_direction", output_dir="annotations", method='max', ratio=0.99):
    os.makedirs(output_dir, exist_ok=True)
    N_range, scores = compute_interclass_scores(df,MAX_N,direction_cols)

    if method == 'knee':
        best_N = plot_knee(N_range, scores, os.path.join(output_dir, "interclass_score_knee.png"))
    elif method == 'max':
        best_N = plot_max(N_range, scores, os.path.join(output_dir, "interclass_score_max.png"))
    elif method == 'near_max':
        best_N = plot_near_max(N_range, scores, os.path.join(output_dir, "interclass_score_near_max.png"), ratio=ratio)
    else:
        raise ValueError("method should be 'knee', 'max', or 'near_max'")

    print(f"✔️ 自动检测最佳方向长度 N = {best_N}")
    return best_N, N_range, scores


def normalize(arr):
    arr = np.array(arr).reshape(-1, 1)
    return MinMaxScaler().fit_transform(arr).flatten()

def moving_average(arr, k):
    return np.convolve(arr, np.ones(k)/k, mode='valid')

def auto_find_best_N_combined(
    df,
    MAX_N,
    direction_cols=["udps.bi_flow_pkt_direction", "udps.bi_pkt_size"],
    alpha=0.5,
    window=5
):
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有字段的原始分数、平滑分数
    raw_scores_dict = {}
    smoothed_scores_dict = {}
    smooth_N_range = None  # 所有字段共用的N范围

    print("✅第三步：开始寻找best_N")
    for col in direction_cols:
        print(f"  🔹 处理字段：{col}")
        N_range, scores = compute_interclass_scores(df, MAX_N, direction_col=col)
        raw_scores_dict[col] = scores
        smoothed_scores = moving_average(scores, window)
        smoothed_scores_dict[col] = smoothed_scores
        if smooth_N_range is None:
            smooth_N_range = N_range[window - 1:]

    # 取前两个字段用于加权融合（如方向、大小）
    if len(direction_cols) < 2:
        raise ValueError("至少需要两个字段用于融合评分")

    col1, col2 = direction_cols[:2]
    score1 = smoothed_scores_dict[col1]
    score2 = smoothed_scores_dict[col2]

    score1_norm = normalize(score1)
    score2_norm = normalize(score2)
    combined_scores = alpha * score1_norm + (1 - alpha) * score2_norm

    smooth_N_range = N_range[window - 1:]

    # 自动估计最大增长阈值：使用前10%的变化量估计噪声水平
    noise_threshold = np.percentile(np.abs(np.diff(combined_scores[:max(3, len(combined_scores)//10)])), 90)
    # 找到从最大值往前第一个“停止增长点”
    max_idx = np.argmax(combined_scores)
    for i in range(max_idx):
        if combined_scores[max_idx] - combined_scores[i] >= noise_threshold:
            continue
        best_idx = i
        break
    else:
        best_idx = max_idx  # 如果一直在增长，就选最大值点

    # best_idx = np.argmax(combined_scores)
    best_N = smooth_N_range[best_idx]

    # 归一化平滑前的分数
    score1_raw_norm = normalize(raw_scores_dict[col1])
    score2_raw_norm = normalize(raw_scores_dict[col2])

    plt.figure(figsize=(10, 6))

    # 绘制平滑前的折线（用较淡的颜色或虚线）
    plt.plot(smooth_N_range, score1_raw_norm[window - 1:], label="Packet Direction (raw norm)", linestyle=':',
             color='blue', alpha=0.5)
    plt.plot(smooth_N_range, score2_raw_norm[window - 1:], label="Packet Size (raw norm)", linestyle=':', color='green',
             alpha=0.5)

    # 绘制平滑后的折线（你已有的）
    plt.plot(smooth_N_range, score1_norm, label="Packet Direction (smoothed norm)", linestyle='--', color='blue')
    plt.plot(smooth_N_range, score2_norm, label="Packet Size (smoothed norm)", linestyle='-.', color='green')
    plt.plot(smooth_N_range, combined_scores, label=f"Combined α={alpha}", color='red')

    plt.axvline(x=best_N, color='black', linestyle=':', label=f"N_best = {best_N}")
    plt.axhline(y=combined_scores[best_idx], color='black', linestyle=':',
                label=f"InterclassGap_N_best = {combined_scores[best_idx]:.3f}")
    plt.scatter(best_N, combined_scores[best_idx], color='red', zorder=5)

    plt.xlabel("N (Number of Packets)")
    # plt.xlim(window, MAX_N)  # 确保横轴从 window 开始
    plt.ylabel("InterClassGap (Normalized)")
    # plt.title("Combined InterClass Score vs N")
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "N_best.png"))
    plt.show()

    return best_N, smooth_N_range, combined_scores[best_idx]



