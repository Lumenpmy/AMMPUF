
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



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

# 每个包最大保留字节数
M_best = 79
N_best =33

# 十六进制转整数列表
def hex_to_int_list(pkt_hex, max_len=M_best):
    try:
        pkt_hex = pkt_hex.strip().replace(' ', '').replace('[', '').replace(']', '').replace(',', '')
        if len(pkt_hex) % 2 != 0:
            pkt_hex += '0'
        byte_list = [int(pkt_hex[i:i+2], 16) for i in range(0, len(pkt_hex), 2)]
        byte_list = byte_list[:max_len]  # 截断
        return byte_list + [0] * (max_len - len(byte_list))  # 填充
    except:
        return [0] * max_len

# 将每个流转换成 [40 * 50] 的长向量（40个包，每个包50字节）
def flow_to_matrix(payload_list):
    if not isinstance(payload_list, list):
        return None
    pkt_mtx = []
    for i in range(N_best):
        if i < len(payload_list):
            pkt_bytes = hex_to_int_list(payload_list[i])
        else:
            pkt_bytes = [0] * M_best  # 空包
        pkt_mtx.append(pkt_bytes)
    return np.array(pkt_mtx).flatten()  # 展平成向量

# 主处理流程
def plot_payload_heatmaps(df, output_dir="heatmap", max_packets=N_best, max_bytes=M_best, min_packets=5):
    os.makedirs(output_dir, exist_ok=True)

    labels = df['label'].unique()
    for label in labels:
        subset = df[df['label'] == label]
        print('labels',len(subset))
        num_flows = len(subset)
        subset = subset.head(num_flows)

        matrix = []
        for payload_list in subset['udps.bi_payload_hex']:
            # 过滤掉数据包数少于 min_packets 的流
            # print(len(payload_list))
            # if len(payload_list) < min_packets:
            #     continue

            flow_matrix = []
            for pkt_hex in payload_list[:max_packets]:
                byte_values = hex_to_int_list(pkt_hex)[:max_bytes]
                byte_values += [0] * (max_bytes - len(byte_values))
                flow_matrix.append(byte_values)

            while len(flow_matrix) < max_packets:
                flow_matrix.append([0] * max_bytes)

            flat_flow = np.concatenate(flow_matrix)
            matrix.append(flat_flow)

        if matrix:  # 只有在有有效流的情况下才画图
            matrix = np.array(matrix)  # shape: [num_flows, max_packets * max_bytes]

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

            img = plt.imshow(matrix, cmap='OrRd', vmin=0, vmax=255, aspect='auto', interpolation='none')

            # 标题 & 标签
            plt.title(f"{label_names.get(label, 'Unknown')} - Payload Heatmap")
            plt.xlabel("Packet Index")  # 包序号
            plt.ylabel("Flow Index")

            # 设置纵坐标（只显示首尾）
            plt.yticks([0, len(matrix) - 1], ['0', str(len(matrix) - 1)])

            # ✅ 设置横坐标：包索引
            tick_interval = 5
            xticks = [i * max_bytes for i in range(0, max_packets, tick_interval)]  # 每个包的起始字节列
            xtick_labels = [str(i) for i in range(0, max_packets, tick_interval)]  # 包编号
            # plt.xticks(ticks=xticks, labels=xtick_labels, rotation=90)

            # plt.xticks(ticks=xticks)  # 设置位置
            # ax = plt.gca()  # 获取当前坐标轴
            # ax.set_xticklabels(xtick_labels)  # 设置标签内容
            #
            # # 手动逐个旋转
            # for label in ax.get_xticklabels():
            #     label.set_rotation(90)

            ax = plt.gca()
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation=0, ha='right')  # 强制设置 rotation + 对齐

            # 添加颜色条
            cbar = plt.colorbar(img)
            cbar.set_label("Byte Value")

            plt.tight_layout()
            safe_label = str(label).replace('/', '_').replace('\\', '_').replace(' ', '_')
            file_path = os.path.join(output_dir, f"payload_heatmap_label_{safe_label}.png")
            plt.savefig(file_path)
            plt.show()




import pandas as pd

# 读取你的CSV
df = pd.read_csv(r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\01_feature_extract\mit\merge_USTC_4_class_hex_data_mit_filtered.csv")

# 如果 payload_hex 是字符串，先转回列表
import ast
df['udps.bi_payload_hex'] = df['udps.bi_payload_hex'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 画图！
plot_payload_heatmaps(df, output_dir="heatmap")