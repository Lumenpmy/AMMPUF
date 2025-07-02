import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import ot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# # vnat
# label_names = {
#     1: "Vimeo", 2: "Netflix", 3: "Youtube",
#     4: "Skype", 5: "Ssh",6:"Rdp",7:"Sftp",8:"Rsync",9:"Scp"
# }


# ZEAC
# label_names = {
#     1: "Chrome(Browse)", 2: "Firefox(Browse)", 3: "Gmail",4: "Ftps",5:"Youtube(Stream)"
# }

# ISCXVPN
# label_names = {
#     1: "Facebook(Chat)", 2: "Hangouts(Chat)", 3: "Skype(Chat)",4: "Email(Email)",5:"Netflix(Stream)",
#     6: "Spotify(Stream)", 7: "Viome(Stream)", 8: "Youtube(Stream)",9: "FTP(File)",10:"Skype(File)",
#     11: "Email(Voip)", 12: "Hangouts(Voip)", 13: "Skype(Voip)",14: "Buster(Voip)"
# }

#USTC
label_names = {
    1: "Gmail", 2: "SMB", 3: "Weibo",4: "WOW"
}
custom_labels = [label_names[i] for i in sorted(label_names.keys())]


N_best=5
M_best=79
df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\01_feature_extract\mit\merge_USTC_4_class_hex_data_mit_filtered.csv")
# label_list = sorted(df['label'].unique())  # 自动提取 label 列中存在的 label
# custom_labels = [str(i) for i in label_list]

# ========== Payload 转矩阵 ==========
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

# ========== direction Wasserstein ==========
def plot_direction_wasserstein_heatmap(df, output_dir, PACKET_LENGTH=N_best):
    os.makedirs(output_dir, exist_ok=True)
    label_avg_vectors = {}

    for label, group in df.groupby('label'):
        sampled = group.sample(n=len(group), random_state=42, replace=False).reset_index(drop=True)
        matrix = np.full((len(sampled), PACKET_LENGTH), -1)

        for i in range(len(sampled)):
            direction_str = sampled.iloc[i]['udps.bi_flow_pkt_direction']
            direction_seq = direction_str.strip().split()[:PACKET_LENGTH]
            direction_array = [int(x) for x in direction_seq]
            matrix[i, :len(direction_array)] = direction_array

        masked_matrix = np.ma.masked_equal(matrix, -1)
        avg_vector = masked_matrix.mean(axis=0).filled(0)
        label_avg_vectors[label] = avg_vector

    label_list = sorted(label_avg_vectors.keys(), key=int)
    w_dist_matrix = np.zeros((len(label_list), len(label_list)))

    for i in range(len(label_list)):
        for j in range(len(label_list)):
            vec1 = label_avg_vectors[label_list[i]]
            vec2 = label_avg_vectors[label_list[j]]
            w_dist_matrix[i, j] = wasserstein_distance(vec1, vec2)

    scaled_matrix = w_dist_matrix / np.max(w_dist_matrix)
    # scaled_matrix = w_dist_matrix
    _plot_heatmap(scaled_matrix, custom_labels,
                  title="Normalized Wasserstein Distance of Avg Packet Direction",
                  output_path=os.path.join(output_dir, "direction_wasserstein_heatmap.png"))


# ========== size Wasserstein==========
def plot_packetsize_wasserstein_heatmap(df, output_dir, PACKET_LENGTH=N_best):
    os.makedirs(output_dir, exist_ok=True)
    label_avg_vectors = {}

    for label, group in df.groupby('label'):
        sampled = group.sample(n=len(group), random_state=42).reset_index(drop=True)
        matrix = np.zeros((len(sampled), PACKET_LENGTH))

        for i in range(len(sampled)):
            packet_size = sampled.iloc[i]['udps.bi_pkt_size']
            direction = sampled.iloc[i]['udps.bi_flow_pkt_direction']

            size_seq = packet_size.strip().split()[:PACKET_LENGTH]
            dir_seq = direction.strip().split()[:PACKET_LENGTH]

            size_array = [int(x) for x in size_seq]
            dir_array = [int(x) for x in dir_seq]

            # 根据方向调整正负
            adjusted_sizes = [s if d == 1 else -s for s, d in zip(size_array, dir_array)]

            matrix[i, :len(size_array)] = adjusted_sizes

        label_avg_vectors[label] = matrix.mean(axis=0)

    label_list = sorted(label_avg_vectors.keys(), key=int)
    w_dist_matrix = np.zeros((len(label_list), len(label_list)))
    for i in range(len(label_list)):
        for j in range(len(label_list)):
            vec1 = label_avg_vectors[label_list[i]]
            vec2 = label_avg_vectors[label_list[j]]
            w_dist_matrix[i, j] = wasserstein_distance(vec1, vec2)

    scaled_matrix = w_dist_matrix / np.max(w_dist_matrix)
    # def normalize(arr):
    #     arr = np.array(arr).reshape(-1, 1)
    #     return MinMaxScaler().fit_transform(arr).flatten()
    # scaled_matrix = normalize(w_dist_matrix)
    _plot_heatmap(scaled_matrix, custom_labels,
                  title="Normalized Wasserstein Distance of Avg Packet size",
                  output_path=os.path.join(output_dir, "size_wasserstein_heatmap.png"))
# ========== Payload Wasserstein==========
def plot_payload_wasserstein_heatmap(df, output_dir, PACKET_LENGTH=N_best, PAYLOAD_LEN=M_best):
    os.makedirs(output_dir, exist_ok=True)
    label_avg_matrices = {}

    for label, group in df.groupby('label'):
        sampled = group.sample(n=len(group), random_state=42, replace=False).reset_index(drop=True)
        matrices = []
        for _, row in sampled.iterrows():
            payload_seq = row['udps.bi_payload_hex']
            mat = payload_to_matrix(payload_seq, PACKET_LENGTH, PAYLOAD_LEN)
            matrices.append(mat)
        avg_matrix = np.mean(matrices, axis=0)
        label_avg_matrices[label] = avg_matrix

    label_list = sorted(label_avg_matrices.keys(), key=int)
    w_dist_matrix = np.zeros((len(label_list), len(label_list)))

    for i in range(len(label_list)):
        for j in range(len(label_list)):
            mat1 = label_avg_matrices[label_list[i]]
            mat2 = label_avg_matrices[label_list[j]]
            a = np.ones((PACKET_LENGTH,)) / PACKET_LENGTH
            b = np.ones((PACKET_LENGTH,)) / PACKET_LENGTH
            cost_matrix = ot.dist(mat1, mat2, metric='euclidean')
            wasserstein = ot.emd2(a, b, cost_matrix)
            w_dist_matrix[i, j] = wasserstein

    scaled_matrix = w_dist_matrix / np.max(w_dist_matrix)
    _plot_heatmap(scaled_matrix, custom_labels,
                  title="Normalized Wasserstein Distance of Avg Payload Matrix",
                  output_path=os.path.join(output_dir, "payload_wasserstein_heatmap.png"))

def _plot_heatmap(scaled_matrix, labels, title, output_path):
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'Times New Roman',
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    })
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(scaled_matrix,
                     xticklabels=labels,
                     yticklabels=labels,
                     cmap='OrRd',
                     annot=True, fmt=".2f",
                     annot_kws={"size": 15})
    # ax.set_title(title)
    # ax.set_xlabel("Class")
    # ax.set_ylabel("Class")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


plot_direction_wasserstein_heatmap(df, output_dir="results")
plot_payload_wasserstein_heatmap(df, output_dir="results")
plot_packetsize_wasserstein_heatmap(df, output_dir="results")

