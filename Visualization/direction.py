
from matplotlib.colors import ListedColormap

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ttest_ind, f_oneway

# 读取CSV文件（替换为实际路径）
df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\01_feature_extract\mit\merge_USTC_4_class_hex_data_mit_filtered.csv")

from matplotlib.colors import ListedColormap, BoundaryNorm

# 三色映射：灰-白-黑
custom_cmap = ListedColormap(['#FFFFFF', '#AAAAAA', '#000000'])  # [-1, 0, 1]
boundaries = [-1.5, -0.5, 0.5, 1.5]
norm = BoundaryNorm(boundaries, custom_cmap.N)

# 参数配置
SAMPLES_PER_CLASS = 20 # 每类显示样本数
PACKET_LENGTH =33 # 显示前40个数据包
# COLOR_MAP = 'Oranges'  # 颜色映射（黑:1，白:0）
# COLOR_MAP = 'OrRd'  # 颜色映射（黑:1，白:0）
# custom_cmap = ListedColormap(['#000000', '#FFFFFF'])
# vnat
# label_names = {
#     1: "Vimeo", 2: "Netflix", 3: "Youtube",
#     4: "Skype", 5: "Ssh",6:"Rdp",7:"Sftp",8:"Rsync",9:"Scp"
# }


# ZEAC
# label_names = {
#     1: "Chrome(Browse)", 2: "Firefox(Browse)", 3: "Gmail",4: "Ftps",5:"Youtube(Stream)"
# }

#ISCXVPN
# label_names = {
#     1: "Facebook(Chat)", 2: "Hangouts(Chat)", 3: "Skype(Chat)",4: "Email(Email)",5:"Netflix(Stream)",
#     6: "Spotify(Stream)", 7: "Viome(Stream)", 8: "Youtube(Stream)",9: "FTP(File)",10:"Skype(File)",
#     11: "Email(Voip)", 12: "Hangouts(Voip)", 13: "Skype(Voip)",14: "Buster(Voip)"
# }

#USTC
label_names = {
    1: "Gmail", 2: "SMB", 3: "Weibo",4: "WOW"
}
def visualize_directions(df,output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
    # 按类别分组
    groups = df.groupby('label')

    for label, group in groups:
        # 随机采样（确保可重复性）
        sampled = group.sample(
            n=max(SAMPLES_PER_CLASS,100),
            random_state=42,
            replace=False
        ).reset_index(drop=True)

        # 创建矩阵（默认空）
        matrix = np.full((len(sampled), PACKET_LENGTH), -1)

        # 填入真实方向
        for i in range(len(sampled)):
            direction_str = sampled.iloc[i]['udps.bi_flow_pkt_direction']
            direction_seq = direction_str.strip().split()[:PACKET_LENGTH]
            direction_array = [int(x) for x in direction_seq]
            matrix[i, :len(direction_array)] = direction_array

        # 创建可视化
        plt.figure(figsize=(15, 5))
        plt.rcParams.update({
            'font.size': 15,
            'font.family': 'Times New Roman',
            'axes.titlesize': 15,
            'axes.labelsize': 15,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 15,
        })
        img = plt.imshow(matrix, cmap=custom_cmap, norm=norm, aspect='auto', interpolation='none')

        # 设置坐标轴
        # plt.title(f"Class {label} - Packet Directions", fontsize=14)
        plt.title(f"{label_names.get(label, 'Unknown')} - Packet Directions")
        plt.xlabel("Packet Sequence")

        plt.ylabel("Flow Index")

        plt.yticks([0, len(sampled) - 1], ['0', str(len(sampled) - 1)])
        # plt.xticks(ticks=range(0, PACKET_LENGTH, 5), labels=[str(i + 1) for i in range(0, PACKET_LENGTH, 5)])

        # 添加颜色条
        cbar = plt.colorbar(img, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['No Packet', 'Down (0)', 'Up (1)'])

        plt.tight_layout()
        safe_label = str(label).replace('/', '_').replace('\\', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"direction_heatmap_label_{safe_label}.png")
        plt.savefig(file_path)
        plt.show()

# # 执行可视化
visualize_directions(df, output_dir="direction")