import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os

# 方法一
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

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

N_best = 36
def visualize_packet_sizes(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录



    COLOR_MAP = sns.color_palette("OrRd", as_cmap=True)



    for label, group in df.groupby('label'):
        # 随机采样
        sampled = group.sample(n=len(group),
                               random_state=42).reset_index(drop=True)

        # 构建数据矩阵
        matrix = np.ones((len(sampled), N_best))
        for i in range(len(sampled)):
            pkt_sizes = list(map(int, sampled.iloc[i]['udps.bi_pkt_size'].split()[:N_best]))

            # 补长度，避免空位为 0 导致 LogNorm 出现灰色
            if len(pkt_sizes) < N_best:
                pkt_sizes += [1] * (N_best- len(pkt_sizes))  # 用1填充而非0

            matrix[i] = pkt_sizes

        # # 替换所有0为1，避免log(0)
        # matrix[matrix == 0] = 1

        # 创建热力图
        plt.figure(figsize=(18, 6))
        ax = sns.heatmap(matrix,
                         cmap=COLOR_MAP,
                         norm=LogNorm(vmin=1, vmax=matrix.max()),
                         linewidths=0.5,
                         linecolor='lightgray',
                         annot=False,
                         cbar_kws={'label': 'Packet Size (Bytes)'})

        # 设置标题
        ax.set_title(f"Class {label} - Packet Size Distribution", fontsize=14)
        ax.set_xlabel("Packet Sequence (1-31)", fontsize=12)
        ax.set_ylabel("Flow Index", fontsize=12)
        ax.set_yticks([0, len(sampled) - 1])
        ax.set_yticklabels(['0', str(len(sampled) - 1)])



        # 添加大包边框标记
        threshold = 1000
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if matrix[y, x] > threshold:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False,
                                               edgecolor='blue', lw=1))

        plt.tight_layout()

        # 保存图像
        safe_label = str(label).replace('/', '_').replace('\\', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"pkt_size_heatmap_label_{safe_label}.png")
        plt.savefig(file_path)
        plt.show()


df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\01_feature_extract\mit\merge_ISCXVPN_14_class_hex_data_mit_filtered.csv")
visualize_packet_sizes(df, output_dir='annotations')