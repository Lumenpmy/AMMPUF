"""
@Created Time : 2022/11/29
@Author  : LiYao
@FileName: NewGraphDataset.py
@Description:图神经网络输入，新构图法
@Modified:
    :First modified
    :Modified content:040312整理上传
"""
import copy
import random

import math
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np

class GraphMitDataset(InMemoryDataset):
    def __init__(self, root, label_dict, node_counts, transform=None, pre_transform=None):
        self.node_counts = node_counts
        self.root = root

        # 将 label1, label2, ..., label15 循环赋值成 self.label1, self.label2 ...
        for label_name, label_value in label_dict.items():
            setattr(self, label_name, label_value)  # 动态赋值 self.label1 = 1 这种

        # 同时存一份 label 列表方便循环
        self.labels = list(label_dict.values())

        # super().__init__(root, transform, pre_transform)


        self.node_counts = node_counts
        self.var_size = 5
        self.src_data = pd.read_csv(r'01_feature_extract\mit\merge_USTC_4_class_hex_data_mit.csv')
        self.break_num = 100000
        # 数据的下载和处理过程在父类中调用实现
        super(GraphMitDataset, self).__init__(root, transform, pre_transform) # 会自动调用process函数
        # 加载数据
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices= torch.load(self.processed_paths[0], weights_only=False)

    # 将函数修饰为类属性
    @property
    def raw_file_names(self):
        return ['file_1', 'file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def node_features(self, floor_num: int, direction: [], pkt_size: [], same_floor_dict: {}):
        """添加节点特征
        同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
        包长度，同层包总长度，上层包总长度，下层包总长度
        包长度同层占比，字节/同层包个数（平均包字节数）
        """
        # 每张图的特征向量
        pic_features_vec = []
        # 每个节点的特征向量
        node_vec = []
        # 同层包个数，上层包个数，下层包个数，同层字节数，上层字节数，下层字节数
        same_floor_num = 0
        last_floor_num = 0
        next_floor_num = 0
        same_floor_size = 0
        last_floor_size = 0
        next_floor_size = 0

        for floor in range(0, floor_num + 1):
            if floor == 0:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                if floor_num != 0:
                    for node_next in same_floor_dict[floor + 1]:
                        next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数
                    next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数

                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''

                # pmy；通过字节数的正负来区分方向，字节数位正：出客户端，方向为1，为负，进入客户端，为0
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    # pmy：保证分母不为0，但是想说为什么跟数据包数不同，要反过来呢？
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            elif floor == floor_num:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                '''添加节点特征
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                包长度，同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            else:
                # 获取该层每个节点的index
                for node_same in same_floor_dict[floor]:
                    same_floor_size += int(pkt_size[node_same])  # 计算该层字节总数
                for node_last in same_floor_dict[floor - 1]:
                    last_floor_size += int(pkt_size[node_last])  # 计算上层字节总数
                for node_next in same_floor_dict[floor + 1]:
                    next_floor_size += int(pkt_size[node_next])  # 计算下层字节总数

                same_floor_num = len(same_floor_dict[floor])  # 计算该层总节点数
                last_floor_num = len(same_floor_dict[floor - 1])  # 上层节点数
                next_floor_num = len(same_floor_dict[floor + 1])  # 下层节点数
                '''添加节点特征
                包长度，
                同层个数，上层突发个数，下层突发个数，上/同层包个数比，下/同层包个数比
                同层包总长度，上层包总长度，下层包总长度
                包长度同层占比，字节/同层包个数（平均包字节数）
                '''
                for node in same_floor_dict[floor]:
                    if direction[node] == '1':
                        node_vec.append(int(pkt_size[node]))
                    elif direction[node] == '0':
                        node_vec.append(0 - int(pkt_size[node]))
                    node_vec.append(same_floor_num)
                    node_vec.append(last_floor_num)
                    node_vec.append(next_floor_num)
                    node_vec.append(last_floor_num / same_floor_num)
                    node_vec.append(next_floor_num / same_floor_num)
                    node_vec.append(same_floor_size)
                    node_vec.append(last_floor_size)
                    node_vec.append(next_floor_size)
                    node_vec.append(int(pkt_size[node]) / same_floor_size)
                    node_vec.append(same_floor_size / same_floor_num)

                    # 0329 add feature
                    if last_floor_size  == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / last_floor_size)
                    if next_floor_size == 0:
                        node_vec.append(0)
                    else:
                        node_vec.append(same_floor_size / next_floor_size)

                    # 添加到特征矩阵中
                    pic_features_vec.append(copy.deepcopy(node_vec))
                    node_vec.clear()
            # 一层循环完后全部归零
            same_floor_num = 0
            last_floor_num = 0
            next_floor_num = 0
            same_floor_size = 0
            last_floor_size = 0
            next_floor_size = 0

        return pic_features_vec

    def generate_new_graph(self, direction_list: [], pkt_size_list: [], label):
        """生成无向图"""
        dir_length = len(direction_list)
        # 生成邻接矩阵存边
        neighbor_matrix = np.zeros((dir_length, dir_length))
        # 突发转换标志位
        change_flag = direction_list[0]
        # 同层字典
        same_floor_dict = {}
        # 层数
        floor_num = 0
        #0329 边属性
        edge_attr_list = []
        # 流切片
        for index in range(0, dir_length):
            # index是节点序号，这个循环等于遍历所有节点
            if index == 0:
                same_floor_dict.update({floor_num: [index]})
                continue
            if direction_list[index] == change_flag:
                same_floor_dict[floor_num].append(index)
            else:
                # 不等于前面一个的标志位说明转换了，层数+1，且标志位需要转换
                floor_num += 1
                change_flag = direction_list[index]
                same_floor_dict.update({floor_num: [index]})

        # 遍历所有层，添加边
        for i in range(0, floor_num + 1):
            # 遍历同层节点
            for seq_index, seq in enumerate(same_floor_dict[i]):
                if seq_index == 0:
                    continue
                neighbor_matrix[seq][seq - 1] = 1
                neighbor_matrix[seq - 1][seq] = 1

            if i == 0:
                continue
            for last_seq in same_floor_dict[i - 1]:
                for this_seq in same_floor_dict[i]:
                    neighbor_matrix[this_seq][last_seq] = 1
                    neighbor_matrix[last_seq][this_seq] = 1

        edge_index = np.argwhere(neighbor_matrix == 1).tolist() # COO（Coordinate List）稀疏图表示
        Y = [int(label)]
        return edge_index, Y, self.node_features(floor_num, direction_list, pkt_size_list, same_floor_dict)

    def process(self):

        data_src = self.src_data
        valid_indices = []
        data_list = []

        num_labels = 4

        for i in range(1, num_labels + 1):
            label_data = data_src.loc[data_src['label'] == i]
            label_indices = label_data.index.tolist()

            pkt_direction_list = label_data['udps.bi_flow_pkt_direction'].tolist()
            pkt_size_list = label_data['udps.bi_pkt_size'].tolist()
            payload_hex_list = label_data['udps.bi_payload_hex'].tolist()

            print(f'label{i} length: {len(pkt_direction_list)}')

            index_cnt = 0
            start_len = len(data_list)

            for index in range(len(pkt_direction_list)):
                # 检查是否为空或非法
                if pkt_direction_list[index] != pkt_direction_list[index]:
                    continue
                if not isinstance(pkt_direction_list[index], str) or len(pkt_direction_list[index].strip()) <= 1:
                    continue

                temp_list = pkt_direction_list[index].split(' ')
                pkt_size_temp = pkt_size_list[index].split(' ')
                temp_list.pop()
                pkt_size_temp.pop()

                if len(temp_list) < self.var_size or (np.array(temp_list[0:self.node_counts]) == '1').all() \
                        or (np.array(temp_list[0:self.node_counts]) == '0').all():
                    continue

                if isinstance(payload_hex_list[index], str) and payload_hex_list[index].strip() == '[]':
                    continue

                temp_list = temp_list[0:self.node_counts]
                pkt_size_temp = pkt_size_temp[0:self.node_counts]

                valid_indices.append(label_indices[index])

                # 动态获取 self.label{i}
                label_attr_name = f'label{i}'
                Y_label = getattr(self, label_attr_name)

                edge_index, Y, x = self.generate_new_graph(temp_list, pkt_size_list=pkt_size_temp, label=Y_label)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)
                x = torch.tensor(x, dtype=torch.float)

                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=Y)
                data_list.append(data)

                index_cnt += 1
                if self.break_num == index_cnt:
                    break

            end_len = len(data_list)
            print(f'label{i} 有 {end_len - start_len} 条数据')


        print('共计' + str(len(data_list)) + '条数据')

        valid_indices_df = pd.DataFrame(valid_indices, columns=['valid_indices'])
        valid_indices_df.to_csv('./interaction_matters_data/graph/ZCX/valid_indices.csv', index=False)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

