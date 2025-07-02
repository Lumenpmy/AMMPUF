import os
import pandas as pd
import numpy as np

from M_best import auto_find_best_M_fixed_N

from N_best import find_best_N_for_85_coverage,auto_find_best_N_combined
def filter(csv_path):

    df=pd.read_csv(csv_path)
    # df=df[df['label']!=10]

    # df=df[df['label']!=10]
    label_pkt_direction = df['udps.bi_flow_pkt_direction'].tolist()
    label_pkt_size = df['udps.bi_pkt_size'].tolist()
    label_payload_hex = df['udps.bi_payload_hex'].tolist()
    flow_features_list = []  # 保存每个流的包特征（13维列表组成的大列表）
    valid_indices = []  # 保存有效流的原索引
    max_node=100
    for index in range(0, len(label_pkt_direction)):
        # 检查方向字段是否有效（排除空值或者异常）
        if label_pkt_direction[index] != label_pkt_direction[index]:
            continue  # 如果出现空值，则直接下一条数据

        temp_list = label_pkt_direction[index].split(' ')
        pkt_size_temp_list = label_pkt_size[index].split(' ')

        temp_list.pop()
        pkt_size_temp_list.pop()
        # pmy:筛选数据包量少于5个的，或者前100个数据包的方向都是1或者都是0的
        # np.array(label1_pkt_direction[0:node_counts]) == 1)：如果都是1的话，会是（True,True ,True ~~~）,那么.all()如果判断都是true的话，就会返回true
        if len(temp_list) < 5 or (np.array(temp_list[0:max_node]) == '1').all() \
                or (np.array(temp_list[0:max_node]) == '0').all():
            continue

        if isinstance(label_payload_hex[index], str) and label_payload_hex[index].strip() == '[]':
            continue
        # 到这里是合法的流了
        valid_indices.append(index)  # 保存原索引

    # 将生成的flow_features_list保存回DataFrame，比如新建一列
    df_valid = df.iloc[valid_indices].reset_index(drop=True)  # 只保留有效的流

    # 构建新文件名
    base, ext = os.path.splitext(csv_path)
    new_path = base + "_filtered" + ext

    # 保存
    df_valid.to_csv(new_path, index=False)

    print(f"✅第一步：过滤处理完成，总有效流量数：{len(df_valid)}")

    return df_valid

# def main():
csv_path = r"D:\paper\企业合作\交互行为图\李尧\小论文\代码\CIBG_encrtypted_traffic_service_classification\01_feature_extract\mit\merge_ZCX_5_class_hex_data_mit_filtered.csv"
# 第一步：根据最大100个数据包筛选掉不合格的流：有效数据包数小于5个、前100个数据包并没有交互、以及payload为空的流
df_valid=filter(csv_path)

# 第二步：选择在[5,100]中覆盖85%流的数据包长度（根据方向去选择）的最小N，作为后续计算best_N的最大的范围
MAX_N = find_best_N_for_85_coverage(df_valid,direction_col='udps.bi_flow_pkt_direction',plot=True,coverage_threshold=0.85)

# 第三步：根据MAX_N结合方向和大小的特征自动选择best_N(由于平滑设置的window为5，可以使得最后数据包的best_N的范围是[5,100]，符合之前诗选的原则，所以这个window也不是超参数)
best_N, N_list, graph_interclassgap = auto_find_best_N_combined(df_valid, MAX_N)

print(f"🎯根据方向和大小选择出的该数据集的best_N = {best_N}")

# # 第四步：根据方向和大小特征给出的best_N利用负载字节选择最合适的前M个字节best_M
best_M, payload_interclassgap = auto_find_best_M_fixed_N(df_valid, best_N, direction_col="udps.bi_payload_hex")
print(f"🎯根据负载选择出的该数据集的best_M = {best_M}")

print(f"payload_score={payload_interclassgap}")
print(f"graph_score={graph_interclassgap}")











