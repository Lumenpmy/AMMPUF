"""
交互动作图+包字节序列
"""
import sys
sys.path.append('.')
import torch
import ast
import random
from collections import Counter,defaultdict
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import datetime
from network import create_model,GraphSAGENetwork
import json
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.utils import to_dense_adj

from NewGraphDataset import GraphMitDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
# from DrawPicture import DrawPic
from sklearn.model_selection import  StratifiedKFold,KFold
import os
from tqdm import tqdm
import tiktoken
#
# def init_distributed_mode():
#     dist.init_process_group(backend='nccl')  # 选择nccl作为后端（适合NVIDIA GPU）
#     torch.cuda.set_device(local_rank)  # 设置每个进程的GPU
#     print(f"Rank {dist.get_rank()} initialized.")

class GraphTokenMainClass(object):
    def __init__(self, data, num_classes, test_data,multi_model,valid_data,flag):
        #self.src_graph_model = SAGEGraphNet(num_classes).to(device)
        self.multi_model = multi_model
        self.data = data  # 训练集
        self.test_data = test_data
        self.valid_data = valid_data
        self.flag=flag
        self.accumulation_step=1
        self.optimizier = torch.optim.Adam(self.multi_model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def train(self):


        self.multi_model.train()

        train_batch_cnt = 0
        for i,data_batch in enumerate(self.data):
            #print('train_batch_cnt '+str(train_batch_cnt))
            train_batch_cnt += 1

            target = data_batch["labels"].to(device)  # 确保标签数据在 GPU 上
            target=target-1

            # 确保所有输入数据都移动到 GPU
            node_features = data_batch["node_features"].to(device)
            adj_matrices = data_batch["adj_matrices"].to(device)
            token_data = data_batch["token_data"].to(device)
            # print(node_features.device)
            # print(adj_matrices.device,)

            out = self.multi_model(node_features, adj_matrices, token_data, self.flag,device)
            # _, predicted = torch.max(out, dim=1)
            # print(predicted)
            # print(target.dtype)
            self.optimizier.zero_grad()
            target = target.long()
            loss = self.criterion(out, target)
            # print(f'batch:{i+1},training_loss:{loss.item():.4f}')
            with open(
                    r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/loss.txt",
                    "a") as f:
                f.write(f'Batch {i + 1}, Training Loss: {loss.item():.4f}\n')

            loss=loss/self.accumulation_step

            if (i+1)%self.accumulation_step==0:

                loss.backward()
                self.optimizier.step()

    @torch.no_grad()
    def predict(self):
        self.multi_model.eval()
        correct = 0
        test_batch_cnt = 0
        for i,data_batch in enumerate(self.valid_data):

            # print('test_batch_cnt '+str(test_batch_cnt))
            test_batch_cnt += 1
            target=data_batch["labels"].to(device)
            target=target-1
            # 确保所有输入数据都移动到 GPU
            node_features = data_batch["node_features"].to(device)
            adj_matrices = data_batch["adj_matrices"].to(device)
            token_data = data_batch["token_data"].to(device)

            out = self.multi_model(node_features, adj_matrices, token_data, self.flag,device)
            pred = out.argmax(dim=1)  # 使用概率最高的类别
            correct += int((pred == target).sum())  # 检查真实标签
            loss = self.criterion(out, target)
            # print(f'batch:{i+1},testing_loss:{loss.item():.4f}')
            with open(
                    r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/loss.txt",
                    "a") as f:
                f.write(f'Batch {i + 1}, Validing Loss: {loss.item():.4f}\n')

        return correct / len(self.valid_data.dataset)

    def save_model(self, model_path):
        torch.save(self.multi_model, model_path)

    
    @torch.no_grad()
    def test(self, model_path):
        self.multi_model = torch.load(model_path,weights_only=False)
        self.multi_model.eval()
        correct = 0
        true_list = []
        pred_list = []
        pred_score = []  # 概率向量，用于画pr roc
        for i,data_batch in enumerate(self.test_data):

            target=data_batch["labels"].to(device)
            target=target-1
            # 确保所有输入数据都移动到 GPU
            node_features = data_batch["node_features"].to(device)
            adj_matrices = data_batch["adj_matrices"].to(device)
            token_data = data_batch["token_data"].to(device)

            out = self.multi_model(node_features, adj_matrices, token_data, self.flag,device)
            pred_score.extend(out.tolist())
            pred = out.argmax(dim=1)  # 使用概率最高的类别
            correct += int((pred == target).sum())  # 检查真实标签
            true_list.append(target.tolist())
            pred_list.append(pred.tolist())
            loss = self.criterion(out, target)
            # print(f'batch:{i + 1},validing_loss:{loss.item():.4f}')
            with open(
                    r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/loss.txt",
                    "a") as f:
                f.write(f'Batch {i + 1}, Testing Loss: {loss.item():.4f}\n')

        return correct / len(self.test_data.dataset), true_list, pred_list, pred_score


class GraphTokenDataset(Dataset):
    def __init__(self, data, slices, token_tensors):
        self.data = data
        self.slices = slices
        self.token_data = token_tensors

        # 计算图的数量
        self.num_graphs = len(slices['y']) - 1

        assert len(self.token_data) == self.num_graphs, "token_data 和 labels 长度不匹配！"

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # 获取当前图的 x 和 edge_index 的切片
        start_x, end_x = self.slices['x'][idx].item(), self.slices['x'][idx + 1].item()
        start_edge, end_edge = self.slices['edge_index'][idx].item(), self.slices['edge_index'][idx + 1].item()

        # 提取该图的节点特征
        node_features = self.data.x[start_x:end_x]

        # 提取该图的邻接矩阵
        edge_index = self.data.edge_index[:, start_edge:end_edge]
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=len(node_features)).squeeze(0)

        # 提取该图的 token_data 和 label
        token = self.token_data[idx]
        label = self.data.y[idx]

        return {
            "node_features": node_features,
            "adj_matrix": adj_matrix,
            "token_data": token,
            "label": label
        }
def major_exp_multi_classify(multimodel,name,flag):
    # init_distributed_mode()
    batch_size = 20
    num_epoch_list =list(range(20,21,10))
    num_counts=41

    for num_epoch in num_epoch_list:
        # 获得图结构数据
        graph_structure = GraphMitDataset(
            root=r'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/interaction_matters_data/graph/VNAT/9class',
            label_dict={f'label{i}': i for i in range(1, 15)},
            node_counts=num_counts)
        #
        # print(len(graph_structure.data.x))
        # print(len(graph_structure.x))
        # print(graph_structure.x.shape[0])
        # print(len(graph_structure.x[1]))
        # print(len(graph_structure.y))

        data_src = pd.read_csv(r'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/01_feature_extract/mit/merge_VNAT_9_class_hex_data_mit.csv')

        # 读取图结构数据考虑的流
        valid_indices_df = pd.read_csv(r'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/interaction_matters_data/graph/VNAT/9class/valid_indices.csv')
        valid_indices = valid_indices_df['valid_indices']
        selected_data = data_src.loc[valid_indices]
        # print(selected_data['label'])
        # print(type(valid_indices[0]))
        #
        # print(valid_indices[:5])
        # # print(data_src.head())
        # print(data_src['label'].loc[valid_indices[:5]].values)
        # print(selected_data['label'].values[:10])
        # # 提取包字节序列
        # token_data= selected_data['udps.bi_payload_tokens']
        # # print(type(token_data))
        #
        # token_data_list = token_data.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        #
        # # 现在 token_data_list 中的每个元素应该是一个 Python 列表
        # token_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in token_data_list]
        hex_data = selected_data['udps.bi_payload_hex']

        # 转换包字节序列为列表
        hex_data_list = hex_data.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # 初始化 tiktoken 编码器
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # 存储最终的 token tensor
        token_tensors = []

        for hex in hex_data_list:
            # 获取前 num_node 个数据包的 hex 序列
            # 如果数据包数量少于 num_node，就直接选取全部
            selected_packet = hex[:num_counts]  # 截取前 num_node 个数据包的 hex 字符串

            truncated_packet=[pkt[:100] for pkt in selected_packet]

            # 将选中的 hex 字符串拼接在一起
            hex_string = ''.join(truncated_packet)  # 拼接字符串

            # 用 tokenizer 进行编码
            token_ids = tokenizer.encode(hex_string)  # 编码为 token ids

            # 将编码结果转化为 tensor
            token_tensor = torch.tensor(token_ids, dtype=torch.long)

            # 将 token tensor 添加到最终列表
            token_tensors.append(token_tensor)
        # print(type(token_tensors))
        # print(token_tensors[0])

        # token_tensors 是一个 list，里面的每个元素都是一个 Tensor
        # print(type(token_tensors))
        # print(type(token_tensors[0]))
        # print(token_tensors[0])

        """
        开始在dataset中就拆开归纳图结构数据
        """
        dataset = GraphTokenDataset(graph_structure.data,graph_structure.slices, token_tensors)
        # # 定义计数器
        # count_node_features = 0
        # count_adj_matrix = 0
        # for i in range(len(dataset)):
        #     sample = dataset[i]  # 访问第一个样本
        #     node_features = sample["node_features"]
        #     adj_matrix = sample["adj_matrix"]
        #     token = sample["token_data"]
        #     label = sample["label"]
        #
        #     #检查节点特征维度
        #     if node_features.shape != torch.Size([20, 13]):
        #         count_node_features += 1
        #         print(node_features.shape)
        #
        #     # 检查邻接矩阵维度
        #     if adj_matrix.shape != torch.Size([20, 20]):
        #         count_adj_matrix += 1
        #         print(adj_matrix.shape)
        #
        # # 输出符合条件的样本数量
        # print(f"不符合节点特征维度 [20, 13] 的样本数: {count_node_features}")
        # print(f"不符合邻接矩阵维度 [20, 20] 的样本数: {count_adj_matrix}")
        labels = [dataset.data.y[i].item() for i in range(dataset.num_graphs)]
        # print('labels',labels)
        # print(len(labels))
        best_overall_acc = 0
        best_model_path = ""

        # 定义十重交叉验证
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        # 交叉验证
        accuracy_list = []
        # print(len(dataset.y))
        # print(len(dataset),len(dataset[0].y))
        accuracy_list = []
        f1_macro_list = []
        true_all = []
        pred_all = []
        pred_scr_all = []
        index_save_path = r'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/split_indices'
        os.makedirs(index_save_path, exist_ok=True)
        split_iter = kf.split(np.arange(len(dataset)), labels)
        for fold, (train_idx, test_idx) in enumerate(
                tqdm(split_iter, total=kf.get_n_splits(), desc="🌟 K-Fold Progress")):
            print(f"\n\n========== Fold {fold + 1} ==========")


            # true_all_close = []
            # pred_all_close = []
            # 用于画roc和pr
            pred_scr_all = []

            # # 根据索引划分训练集和验证集
            # train_subset = Subset(dataset, train_idx)
            # test_subset = Subset(dataset, test_idx)
            #
            # # 再次划分训练集 90% 用于训练，10% 作为验证集
            # val_size = int(0.1 * len(train_subset))
            # # max_train_samples=20
            # # train_size=min(max_train_samples,len(train_subset)-val_size)
            #
            # train_size = len(train_subset) - val_size
            # # print('train_size',train_size)
            # # print('val_size',val_size)
            #
            # # train_data, valid_data = torch.utils.data.random_split(train_subset, [train_size, val_size])
            # generator = torch.Generator().manual_seed(42)
            # train_data, valid_data = torch.utils.data.random_split(train_subset, [train_size, val_size],
            #                                                        generator=generator)
            #
            # print('len(train_subset)', len(train_data))
            # print('len(test_subset)', len(test_subset))
            # print('len(valid_data',len(valid_data))
            label_to_indices = defaultdict(list)
            for idx in train_idx:
                label = labels[idx]
                label_to_indices[label].append(idx)

            max_per_class =5000
            selected_train_indices = []
            for label, idx_list in label_to_indices.items():
                if len(idx_list) > max_per_class:
                    selected = random.sample(idx_list, max_per_class)
                else:
                    selected = idx_list
                selected_train_indices.extend(selected)

            val_ratio = 0.1
            train_size = int((1 - val_ratio) * len(selected_train_indices))
            val_size = len(selected_train_indices) - train_size

            generator = torch.Generator().manual_seed(42)
            train_data, valid_data = torch.utils.data.random_split(
                Subset(dataset, selected_train_indices),
                [train_size, val_size],
                generator=generator
            )

            test_subset = Subset(dataset, test_idx)

            for i,sample in enumerate(train_data):
                if sample['token_data'].shape[0]==0:
                    print('------------------------------------token kong-------------------------------',sample['token_data'])

            def count_labels(data, name):
                # 先取出所有标签
                labels = [sample["label"] for sample in data]
                # 统计每个类别出现的次数
                counts = Counter(labels)

                # 打印汇总信息
                total = len(labels)
                print(f"\n{name} 集中各类别样本数量（共 {total} 条）：")
                # for label, count in sorted(counts.items()):
                #     print(f"  类别 {label}: {count} 条")

            # 放在 train/val/test 划分之后
            count_labels(train_data, "训练")
            count_labels(valid_data, "验证")
            count_labels(test_subset, "测试")

            # 获取原始 data_src 索引（这些都是 valid_indices 中的子集）
            train_index_in_data_src = valid_indices.iloc[train_data.indices].tolist()
            val_index_in_data_src = valid_indices.iloc[valid_data.indices].tolist()
            test_index_in_data_src = valid_indices.iloc[test_idx].tolist()

            # 保存为 JSON
            split_record = {
                "fold": fold,
                "train_indices": train_index_in_data_src,
                "val_indices": val_index_in_data_src,
                "test_indices": test_index_in_data_src
            }
            with open(os.path.join(index_save_path, f"fold_{fold}_indices.json"), "w") as f:
                json.dump(split_record, f, indent=2)
            #创建数据加载器
            def collate_fn1(batch):
                # print("Received batch with size:", len(batch))
                node_features_list = [data["node_features"] for data in batch]
                adj_matrices_list = [data["adj_matrix"] for data in batch]
                labels = torch.tensor([data["label"] for data in batch], dtype=torch.long)

                # print("Batch size:", len(batch))
                # for i, (nf, adj) in enumerate(zip(node_features_list, adj_matrices_list)):
                #     print(f"Sample {i}: node_features shape = {nf.shape}, adj_matrix shape = {adj.shape}")

                # **填充 token_data**
                token_data_list = [data["token_data"] for data in batch]
                token_data_padded = pad_sequence(token_data_list, batch_first=True, padding_value=0)
                # print(f"Token data padded shape: {token_data_padded.shape}")

                # **计算最大节点数，并填充邻接矩阵和节点特征**
                max_nodes = max(nf.shape[0] for nf in node_features_list)

                # 创建填充后的 batch
                node_padded = torch.zeros(len(node_features_list), max_nodes, node_features_list[0].shape[-1])
                adj_padded = torch.zeros(len(adj_matrices_list), max_nodes, max_nodes)

                for i, (node_features, adj_matrix) in enumerate(zip(node_features_list, adj_matrices_list)):
                    num_nodes = node_features.shape[0]
                    node_padded[i, :num_nodes, :] = node_features
                    adj_padded[i, :num_nodes, :num_nodes] = adj_matrix
                # 节点不够的部分填充0

                # print(f"Node features padded shape: {node_padded.shape}")
                # print(f"Adjacency matrix padded shape: {adj_padded.shape}")
                # print(f"Token data padded shape: {token_data_padded.shape}")

                return {
                    "node_features": node_padded,
                    "adj_matrices": adj_padded,
                    "token_data": token_data_padded,
                    "labels": labels
                }

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)


            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn1,worker_init_fn=seed_worker,generator=generator)
            # i=0
            # for batch_idx, batch in enumerate(train_loader):
            #
            #         # print(f"Batch {batch_idx + 1}:")
            #
            #         # 打印每个样本的形状
            #         # print(f"Node features: {batch['node_features'][0]}")
            #         # print(f"Adjacency matrices shape: {batch['adj_matrices'].shape}")
            #     print(f"Token data shape: {batch['token_data']}")
            #     print(f"Labels shape: {batch['labels']}")
            #     i+=1
            # print(i)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn1)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn1)
            # for i,batch in enumerate(test_loader):
            #     if i==0:
            #         print(f"Node features: {batch['node_features'][0]}")
            # train_sampler = DistributedSampler(train_data)
            # valid_sampler = DistributedSampler(valid_data, shuffle=False)
            # test_sampler = DistributedSampler(test_subset, shuffle=False)
            #
            # train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn1)
            # valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, collate_fn=collate_fn1)
            # test_loader = DataLoader(test_subset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn1)

            # # 建立网络
            g_obj = GraphTokenMainClass(train_loader, 5, test_loader,multimodel,valid_loader,flag )

            # 训练以及测试
            best_acc = 0
            best_fold_model_path = ""

            epoch_bar = tqdm(range(num_epoch), desc=f"🌀 Training Fold {fold + 1}", leave=False)
            with open(
                    r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/loss.txt",
                    "a") as f:
                f.write(f'Fols{fold}\n')

            patience = 10  # 连续多少个 epoch 验证精度不提升就停止
            early_stop_counter = 0
            for epoch in epoch_bar:
                with open(
                        r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/loss.txt",
                        "a") as f:
                    f.write(f'Epoch{epoch}\n')
                # print("--------正在进行第----------"+str(epoch)+"次训练")
                g_obj.train()
                valid_acc = g_obj.predict()
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    early_stop_counter = 0  # 重置计数器
                    best_fold_model_path = f'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/model/VNAT/9class/{name}_{fold}.pt'
                    # 保存当前折的最佳模型
                    g_obj.save_model(best_fold_model_path)
                    print(f"Saved best model for fold {fold + 1} with accuracy: {best_acc:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"No improvement for {early_stop_counter} epoch(s).")

                    # 如果连续若干轮验证精度没有提升，则提前停止训练
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                    break

            # 比较每一折的最佳准确率，找出最优的一折
            if best_acc > best_overall_acc:
                best_overall_acc = best_acc
                best_model_path = best_fold_model_path  # 更新最优模型路径
            # 最后，输出10折交叉验证中最优的模型和准确率
            print(
                f"\n\n🌟 Best model is from fold {best_model_path.split('_')[-2]} with accuracy: {best_overall_acc:.4f}")

            # 保存最终最优模型
            print(f"Saving the best overall model to {best_model_path}")

            pred_acc, true_list, pred_list, pred_scr = g_obj.test(f'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/model/VNAT/9class/{name}_{fold}.pt')
            # print(pred_acc, (true_list[0][0]), (pred_list[0][0]), pred_scr[0])
            # 1.0 3 3[-58.08167266845703, -81.5223388671875, -69.7474594116211, 0.0, -63.356449127197266]

            true_list = [i for item in true_list for i in item]
            pred_list = [i for item in pred_list for i in item]
            true_all.extend(true_list)
            pred_all.extend(pred_list)
            pred_scr_all.extend(pred_scr)

        # 获取分类报告（每个类别的指标）
        conf_matrix = confusion_matrix(true_all, pred_all)
        np.savetxt(r"/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/confusion_matrix.txt",conf_matrix,fmt="%d")
        FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # 假阳性
        TN = conf_matrix.sum() - (FP + conf_matrix.sum(axis=1) - np.diag(conf_matrix) + np.diag(conf_matrix))  # 真阴性

        report_dict = classification_report(true_all, pred_all, digits=5, output_dict=True)

        # # 2. 添加 overall accuracy 到字典里（不是每类的，而是总体）
        # overall_acc = accuracy_score(true_all, pred_all)
        # report_dict['overall_accuracy'] = {'accuracy': overall_acc}

        class_fprs = []
        # 计算每个类别的 accuracy
        class_accuracies = {}
        num_classes = len(conf_matrix)
        for i in range(num_classes):
            class_acc = conf_matrix[i, i] / conf_matrix[i, :].sum()
            class_fpr = FP[i] / (FP[i] + TN[i] + 1e-6)  # 避免除0
            # report_dict[str(i)]['accuracy'] = class_acc
            report_dict[str(i)]['fpr'] = class_fpr
            class_fprs.append(class_fpr)

        overall_acc = accuracy_score(true_all, pred_all)
        overall_prec = precision_score(true_all, pred_all, average='micro')
        overall_recall = recall_score(true_all, pred_all, average='micro')
        overall_f1 = f1_score(true_all, pred_all, average='micro')

        report_dict['overall'] = {
            'precision': overall_prec,
            'recall': overall_recall,
            'f1-score': overall_f1,
            'accuracy': overall_acc,
            'support': len(true_all)
        }

        # 用混淆矩阵重新计算每类的准确率，取宏平均
        class_accuracies = [conf_matrix[i, i] / (conf_matrix[i, :].sum() + 1e-6) for i in range(num_classes)]
        macro_accuracy = sum(class_accuracies) / num_classes
        report_dict['macro_accuracy'] = {'accuracy': macro_accuracy}
        # 每类样本数
        supports = np.array([report_dict[str(i)]['support'] for i in range(num_classes)])
        # 加权平均 accuracy
        weighted_accuracy = np.sum(np.array(class_accuracies) * supports) / np.sum(supports)
        report_dict['weighted_accuracy'] = {'accuracy': weighted_accuracy}

        FP_total = FP.sum()
        TN_total = TN.sum()
        overall_fpr = FP_total / (FP_total + TN_total + 1e-6)
        report_dict['overall']['fpr'] = overall_fpr
        macro_fpr = np.mean(class_fprs)
        weighted_fpr = np.sum(np.array(class_fprs) * supports) / np.sum(supports)
        report_dict['macro_fpr'] = {'fpr': macro_fpr}
        report_dict['weighted_fpr'] = {'fpr': weighted_fpr}
        # 转换成 DataFrame 并保存
        class_df = pd.DataFrame(report_dict).transpose()
        class_output_csv_path = f'/home/phc/pmy/PycharmProjects/CIBG_encrtypted_traffic_service_classification/02_classification/实验结果/VNAT/9class/{name}_classification_report.csv'
        # class_df.to_csv(class_output_csv_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        class_df.insert(0, 'model_name', name)
        class_df.insert(1, 'timestamp', timestamp)
        class_df.insert(2, 'epoch', num_epoch)

        # 追加一个空行（分隔上次的记录）
        with open(class_output_csv_path, 'a', encoding='utf-8') as f:
            f.write('\n')  # 可选：写入空行作为分隔

        # 写入CSV，使用追加模式，始终保留 header
        class_df.to_csv(class_output_csv_path, mode='a', header=True, index=True)




    return true_all,pred_all


if __name__ == '__main__':
    # torch.cuda.empty_cache()

    #
    # local_rank = int(os.environ["LOCAL_RANK"])  # 获取进程的 local_rank
    # torch.cuda.set_device(local_rank)  # 让当前进程使用 local_rank 对应的 GPU
    # device = torch.device(f"cuda:{local_rank}")
    # device = 'cpu'SP
    # draw_obj = DrawPic()

    # different_graph_model()
    # flag来控制
    # flag来控制

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 确保结果可重复
        # torch.use_deterministic_algorithms(True)


    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(os.getenv("CUBLAS_WORKSPACE_CONFIG"))
    #device=torch.device("cpu")
    print(device)

    print(torch.cuda.device_count())

    # multimodel = create_model(device)

    # #ISCXVPN
    # interclassgap_token = 0.23862391757208523
    # interclassgap_graph = 0.910086862395072
    #VNAT
    interclassgap_token = 0.42514073437876676
    interclassgap_graph = 0.6962073720100967
    # #ISCXVPN
    # interclassgap_token = 0.2680087036605987
    # interclassgap_graph = 0.8937181233957291
    # 
    # # ISCXVPN
    # interclassgap_token = 0.188403141179859
    # interclassgap_graph = 0.6426967860087783


    multimodel = create_model(device, interclassgap_graph, interclassgap_token)
    multimodel=torch.nn.DataParallel(multimodel, device_ids=range(torch.cuda.device_count()))
    flag_map = {
        'GraphSAGE_': 0,
        # 'GraphSAGE_Transformer_': 1,
        # 'Transformer':2

    }

    for model_name in flag_map:
        major_exp_multi_classify(multimodel.to(device),name=model_name, flag=flag_map[model_name])
