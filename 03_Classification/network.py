import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import math
import tiktoken

class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)
        
    def forward(self, x, adj_matrix):
        # 计算邻居节点的聚合特征
        neighbor_features = torch.matmul(adj_matrix, x)
        # 将节点自身特征和邻居特征拼接
        combined = torch.cat([x, neighbor_features], dim=-1)
        # 通过线性层和激活函数
        out = self.linear(combined)
        return F.gelu(out)

class GraphSAGENetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENetwork, self).__init__()
        # 具体多少层看效果怎么样吧，我的论文里面用了三层
        self.layer1 = GraphSAGELayer(in_channels, hidden_channels)
        self.layer2 = GraphSAGELayer(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.025)
        
    def forward(self, x, adj_matrix):
        x = self.layer1(x, adj_matrix)
        x = self.dropout(x)
        x = self.layer2(x, adj_matrix)
        return x

class FlashMultiHeadAttention(nn.Module):
    def __init__(self,dropout_rate):
        super(FlashMultiHeadAttention,self).__init__()

        self.d_model = Params.d_model
        self.head_count = Params.head_count

        self.head_dim = self.d_model//self.head_count
        # input shape: [batch,sentence_length,d_model],output shape [batch,sentence_length,d_model*3]
        # 对应计算出来所有token的q k v
        self.weight_matrix_generate = nn.Linear(self.d_model,self.d_model*3)
        # 注意力得分的softmax
        self.softmax = nn.Softmax(-1)
        # input shape: [batch,sentence_length,d_model],output shape [batch,sentence_length,d_model]
        # attention block最后一层，全连接层
        self.out_weight_matrix = nn.Linear(self.d_model,self.d_model) # 融合多个头的信息
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x,padding_mask=None):
        """这是我之前写的代码，如果没有torch2.2以上，则没有flash attention,而是传统的自注意力模型Scaled Dot-Product Attention（SDPA），需要使用我注释掉的代码"""
        # MHA我是自定义的，所以没用pytorch自带的
        batch, sentence_length,_ = x.shape
        qkv = self.weight_matrix_generate(x)
        # 沿着d_model维度切分成三个，就是对应线性变换后的qkv矩阵
        q,k,v = qkv.chunk(3,dim=-1)

        # 多头注意力变形，沿着q k v最后一个维度做头的切割
        # 交换序列长度和头数量（多头内每个头做自己的计算）
        # input shape: [batch,sentence_length,d_model],output shape [batch,head_count,sentence_length,d_model]
        multi_head_q = torch.reshape(q,[batch,sentence_length,self.head_count,self.head_dim]).transpose(1,2).contiguous()
        multi_head_k = torch.reshape(k, [batch, sentence_length, self.head_count, self.head_dim]).transpose(1,2).contiguous()
        multi_head_v = torch.reshape(v, [batch, sentence_length, self.head_count, self.head_dim]).transpose(1,2).contiguous()


        # 以下为不加任何优化的Multi-head Attention
        # input shape [batch,head_count,sentence_length,d_model] output shape [batch,head_count,sentence_length,sentence_length]
        #q_k_scores = multi_head_q.matmul(multi_head_k.transpose(2,3))/self.d_model**0.5 # 计算注意力分数

        # pmy:将padding mask扩展成（batch，1,1,sentence_length）
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
            padding_mask = padding_mask.unsqueeze(1)
            #q_k_scores = q_k_scores.masked_fill(padding_mask==0,float('-inf')) # 需要手动的将False变为-inf，表示被mask

        # 调用flash attention，仅torch2.2以上支持
        # flash attention 做了MHA的q k v矩阵乘法和缩放
        attention_output = F.scaled_dot_product_attention(multi_head_q,multi_head_k,multi_head_v,attn_mask=padding_mask)
        # 但是F.scaled_dot_product_attention不需要自己手动计算注意力分数和自己mask_fill就能得到输出

        # 计算注意力分数
        # softmax_q_k_scores = self.softmax(q_k_scores)
        # attention_scores*v
        # attention_output = softmax_q_k_scores.matmul(multi_head_v)
        # attention_output = self.dropout(attention_output)
        # 合并所有的头
        # input shape:[batch,head_count,sentence_length,d_model],output shape [batch,sentence_length,d_model]
        final_attention_output = attention_output.transpose(1,2).contiguous().reshape([batch, sentence_length,self.d_model])

        output = self.out_weight_matrix(final_attention_output) # pmy:final_attention_output的结果只是拼接，全连接层是让这些头相互交互，学习最优的线性组合
        return output


class Blocks(nn.Module):
    def __init__(self,dropout_rate):
        super(Blocks,self).__init__()

        self.norm_layer1 = nn.RMSNorm(Params.d_model)
        # self.norm_layer1 = nn.LayerNorm(Params.d_model)
        # pmy:RMSNorm（Root Mean Square Normalization），它是一种变体的层归一化方法，比标准的 LayerNorm 计算更高效，并且对 Transformer 模型的稳定性更好
        self.attention_layer = FlashMultiHeadAttention(dropout_rate)
        self.norm_layer2 = nn.RMSNorm(Params.d_model)
        # self.norm_layer2 = nn.LayerNorm(Params.d_model)
        # 前馈隐藏层，最后的linear就是一个block的output
        # input shape: [batch,sentence_length,d_model],output shape [batch,head_count,sentence_length,d_model]
        self.ffn = nn.Sequential(
            nn.Linear(Params.d_model,Params.ffn_size),
            nn.GELU(),
            # nn.Dropout(dropout_rate),  # 在激活后加入 dropout
            nn.Linear(Params.ffn_size, Params.d_model),
            # nn.Dropout(dropout_rate)  # 在输出后加入 dropout
        )

    def forward(self,x,padding_mask = None):
        # block中的两个残差连接（residual connection）
        # pmy：采用pre-LN
        x = x + self.attention_layer(self.norm_layer1(x),padding_mask)
        x = x + self.ffn(self.norm_layer2(x))
        return x

class ModelNetwork(nn.Module):
    def __init__(self,device,dropout_rate=0.025):
        super(ModelNetwork,self).__init__()
        # input size [batch,sentence_len]
        # output size  [batch,sentence_len,d_model] 返回的就是embedding向量，相当于把买个token换成d.model维度的向量
        self.embedding = nn.Embedding(Params.vocab_size,Params.d_model)
        # pmy:加入位置编码
        self.positional_encoding = self._get_positional_encoding(Params.d_model,Params.max_len).to(device)
        print("device",self.positional_encoding.device)
        # pmy:利用tokenizer给每个token一个d.model512维度的向量
        # 定义有多少transformer块
        self.blocks = nn.ModuleList([Blocks(dropout_rate) for _ in range(Params.block_num)])

    def _get_positional_encoding(self, d_model,max_len):
        # 位置编码还是要的，但是不需要RoPE旋转位置编码了，那玩意是搞外推性的，我们这里不需要
        # 只需要一个绝对位置编码就行了，采用transformer原论文的绝对位置编码
        # 生成位置编码（假设是固定的sin/cos位置编码）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加batch维度
        return pe
    def forward(self,x,padding_mask=None):
        # x size: [batch,sentence_length]
        x = self.embedding(x)
        # print("x.device",x.device)
        self.positional_encoding=self.positional_encoding.to(x.device)
        # print(self.positional_encoding.device)
        # pmy
        if x.size(1)>self.positional_encoding.size(1):
            x=x[:,:self.positional_encoding.size(1),:]
        x = x + self.positional_encoding[:, :x.size(1)]

        for block in self.blocks:
            x = block(x,padding_mask)

        return x
   

class NetworkTrafficClassifier(nn.Module):
    def __init__(self, 
                 graph_in_channels,
                 graph_hidden_channels,
                 graph_out_channels,
                 num_classes,device,interclassgap_graph, interclassgap_token):
        super(NetworkTrafficClassifier, self).__init__()


        self.interclassgap_graph=interclassgap_graph
        self.interclassgap_token=interclassgap_token
        
        # GraphSAGE
        self.graph_net = GraphSAGENetwork(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden_channels,
            out_channels=graph_out_channels
        )
        self.graph_mapper = nn.Sequential(
            nn.Linear(graph_out_channels * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.token_mapper = nn.Sequential(
            nn.Linear(Params.d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.gating_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出两个融合权重
        )
        # 编码器
        self.transformer = ModelNetwork(device)
        # 模态对齐层
        self.alignment = nn.Sequential(
            nn.Linear(graph_out_channels * 2 + Params.d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.025),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.025)
        )

        self.alignment_only_sage = nn.Sequential(
            nn.Linear(graph_out_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.025),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.025)
        )

        self.alignment_only_encoder = nn.Sequential(
            nn.Linear(Params.d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.025),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.025)
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, node_features, adj_matrix, token_data,flag,device):
       
            graph_features = self.graph_net(node_features, adj_matrix)

            # 全局池化 取最大池化和平均池化
            graph_max = torch.max(graph_features, dim=1)[0]
            # print('max',graph_max.shape)
            graph_mean = torch.mean(graph_features, dim=1)
            # print('mean',graph_mean.shape)
            graph_features = torch.cat([graph_max, graph_mean],dim=1) # (batch_size, graph_out_channels * 2)
            
            token_features = self.transformer(token_data)
            # 全局池化 取最大池化和平均池化
            token_max = torch.max(token_features, dim=1)[0]
            token_mean = torch.mean(token_features, dim=1)
            token_features = torch.cat([token_max, token_mean], dim=1) # (batch_size, token_out_dim * 2)
        
            graph_features = self.graph_mapper(graph_features)  # [B, fusion_dim]
            token_features = self.token_mapper(token_features)  # [B, fusion_dim]
            # graph_features = self.alignment_only_sage(graph_features)
            # token_features = self.alignment_only_encoder(token_features)

            score_vec = torch.tensor(
                [self.interclassgap_graph, self.interclassgap_token], dtype=torch.float32,
                device=graph_features.device
            ).unsqueeze(0).repeat(graph_features.size(0), 1)  # shape: [B, 2]

            #  输入 gating 网络，得到融合权重
            gate_logits = self.gating_net(score_vec)  # [B, 2]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [B, 2]
            w_g = gate_weights[:, 0].unsqueeze(-1)  # [B, 1]
            w_t = gate_weights[:, 1].unsqueeze(-1)  # [B, 1]

            # 4. 融合特征
            aligned_features = w_g * graph_features + w_t * token_features  # [B, fusion_dim]
        
        # 分类
        output = self.classifier(aligned_features)
        return output

def create_model(device,interclassgap_graph, interclassgap_token,graph_in_channels=13,
                graph_hidden_channels=128,
                graph_out_channels=64,
                num_classes=4):

    model = NetworkTrafficClassifier(
        graph_in_channels=graph_in_channels,
        graph_hidden_channels=graph_hidden_channels,
        graph_out_channels=graph_out_channels,
        num_classes=num_classes,device=device,interclassgap_graph=interclassgap_graph, interclassgap_token=interclassgap_token
    )
    return model

class Params():
    d_model = 512
    head_count = 8
    ffn_size = 4096
    block_num = 5
    max_len=5000
    # tiktoken cl100k_base vocab size
    vocab_size = 100277
    # tiktoken p50k_base vocab size
    # vocab_size = 50281
    # classification_nums = 10
