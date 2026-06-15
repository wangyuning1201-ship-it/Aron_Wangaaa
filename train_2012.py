import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, accuracy_score
from untils import create_ip_mapping_2012, create_graph_data_2012, GraphDataset
from network import ROEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch_geometric.loader import DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim

data1 = pd.read_csv("data/ISCXIDS2012/TestbedMonJun14Flows.csv")
data2 = pd.read_csv("data/ISCXIDS2012/TestbedSatJun12Flows.csv")
data3 = pd.read_csv("data/ISCXIDS2012/TestbedSunJun13Flows.csv")
# data4 = pd.read_csv("data/ISCXIDS2012/TestbedThuJun17Flows.csv")
# data5 = pd.read_csv("data/ISCXIDS2012/TestbedTueJun15Flows.csv")
# data6 = pd.read_csv("data/ISCXIDS2012/TestbedWedJun16Flows.csv")

# 合并所有数据帧
data_list = [data1, data2, data3]
data = pd.concat(data_list, ignore_index=True)

# 解析时间戳
data['generated'] = pd.to_datetime(data['generated'])
data['startDateTime'] = pd.to_datetime(data['startDateTime'])
data['stopDateTime'] = pd.to_datetime(data['stopDateTime'])

# 计算每个会话的持续时间（分钟）
data['Duration of time'] = data['stopDateTime'] - data['startDateTime']
data['Duration of time'] = data['Duration of time'].dt.total_seconds() / 60
data.drop(columns=['stopDateTime', 'startDateTime'], inplace=True)

# 填充缺失值
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# 计算每列的非空值比率
non_null_ratio = data.notna().mean()

# 选择非空值比率小于等于0.3的列并删除
columns_to_drop = non_null_ratio[non_null_ratio <= 0.3].index
data.drop(columns=columns_to_drop, inplace=True)

# 对TCP标志描述进行One-hot编码
encoder = OneHotEncoder()
label_encoder = LabelEncoder()
tcp_flags = data[['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']].fillna('')
encoded_tcp_flags = encoder.fit_transform(tcp_flags)
encoded_tcp_flags_df = pd.DataFrame(encoded_tcp_flags.toarray(), columns=encoder.get_feature_names_out(
    ['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']))
data.drop(columns=['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription'], inplace=True)
data = pd.concat([data, encoded_tcp_flags_df], axis=1)

# 对协议名称进行标签编码
data['protocolName'] = label_encoder.fit_transform(data['protocolName'])
data['sourcePayloadAsUTF'] = label_encoder.fit_transform(data['sourcePayloadAsUTF'])
data['destinationPayloadAsUTF'] = label_encoder.fit_transform(data['destinationPayloadAsUTF'])

# 对方向进行标签编码
data['direction'] = label_encoder.fit_transform(data['direction'])

# 处理负载数据，这里以长度为例
data['sourcePayloadLength'] = data['sourcePayloadAsBase64'].apply(len)
data['destinationPayloadLength'] = data['destinationPayloadAsBase64'].apply(len)
data.drop(columns=['sourcePayloadAsBase64', 'destinationPayloadAsBase64'], inplace=True)

# 按分钟创建时间窗口
data['generated'] = data['generated'].dt.floor('T')

# 按时间窗口分组数据
grouped_data = data.groupby('generated')

# 为每个时间窗口分配动态IP映射并动态构建图
graph_data_seq = []
for name, group in grouped_data:
    ip_mapping = create_ip_mapping_2012(group)
    # 使用半监督学习，只标记20%的边
    graph_data = create_graph_data_2012(group, ip_mapping, time_window=name, labeled_ratio=0.2)
    if graph_data is not None:
        graph_data_seq.append(graph_data)

# 初始化设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将数据集分为训练集和测试集
train_data_seq, test_data_seq = train_test_split(graph_data_seq, test_size=0.2, random_state=42)

# 为训练集和测试集创建DataLoader
train_dataset = GraphDataset(train_data_seq, device=device)
test_dataset = GraphDataset(test_data_seq, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)


def evaluate(model, dataloader):
    """评估模型性能，考虑半监督学习场景"""
    model.eval()  # 将模型设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_preds = []
    all_labeled_mask = []  # 存储哪些边有标签

    with torch.no_grad():  # 关闭梯度计算
        for graph_data in dataloader:
            graph_data = graph_data.to(device)

            # 获取模型预测
            edge_predictions, _ = model(graphs=[graph_data], seq_len=1, train_mode=False)
            edge_labels_batch = graph_data.edge_labels.to(device)
            labeled_mask = graph_data.labeled_mask.to(device)  # 获取标签掩码

            # 获取每条边的预测类别
            _, predicted = torch.max(edge_predictions[0], dim=1)

            # 存储真实标签和预测结果
            all_labels.extend(edge_labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labeled_mask.extend(labeled_mask.cpu().numpy())

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_labeled_mask = np.array(all_labeled_mask)

    # 只计算有标签边的指标
    labeled_indices = np.where(all_labeled_mask == 1)[0]
    labeled_labels = all_labels[labeled_indices]
    labeled_preds = all_preds[labeled_indices]

    if len(labeled_indices) > 0:
        # 计算准确率
        accuracy = 100 * accuracy_score(labeled_labels, labeled_preds)

        # 计算召回率
        recall = recall_score(labeled_labels, labeled_preds, average='weighted')

        # 计算F1分数
        f1 = f1_score(labeled_labels, labeled_preds, average='weighted')

        # 计算精确率
        precision = 100 * precision_score(labeled_labels, labeled_preds, average='weighted')

        # 计算AUC值（对于多分类任务，需要对标签进行One-Hot编码）
        try:
            auc = roc_auc_score(labeled_labels, labeled_preds, multi_class='ovo')
        except ValueError:
            auc = float('nan')  # 如果无法计算AUC，返回NaN
    else:
        accuracy, precision, recall, f1, auc = 0, 0, 0, 0, 0

    model.train()  # 将模型恢复为训练模式

    # 返回评估指标和混淆矩阵值
    return accuracy, precision, recall, f1, auc


def train(model, train_dataloader, test_dataloader, optimizer, criterion, num_epochs, eval_interval, save_dir):
    """训练模型，使用半监督学习"""
    model.train()  # 将模型设置为训练模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 存储训练历史
    train_history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    for epoch in range(num_epochs):
        total_loss = 0
        total_labeled_loss = 0
        total_pseudo_loss = 0
        num_pseudo_labels = 0

        # 训练阶段
        for graph_data in train_dataloader:  # 遍历训练数据
            graph_data = graph_data.to(device)

            optimizer.zero_grad()

            # 获取预测结果和伪标签
            edge_predictions, pseudo_labels_list = model(graphs=[graph_data], seq_len=1, train_mode=True)

            # 获取当前批次的边标签
            edge_labels_batch = graph_data.edge_labels.to(device)
            labeled_mask = graph_data.labeled_mask.to(device)  # 获取标签掩码

            # 计算有标签边的损失
            labeled_loss = criterion(edge_predictions[0][labeled_mask], edge_labels_batch[labeled_mask])

            # 计算伪标签损失（如果有）
            pseudo_loss = 0
            if pseudo_labels_list and len(pseudo_labels_list[0]['pseudo_labels']) > 0:
                pseudo_data = pseudo_labels_list[0]
                pseudo_mask = pseudo_data['pseudo_mask']
                pseudo_labels = pseudo_data['pseudo_labels']

                # 只对高置信度的伪标签计算损失
                pseudo_loss = criterion(edge_predictions[0][pseudo_mask], pseudo_labels)
                num_pseudo_labels += len(pseudo_labels)

            # 总损失 = 有标签损失 + 伪标签损失（加权）
            loss = labeled_loss + 0.5 * pseudo_loss  # 伪标签损失权重设为0.5

            # 反向传播
            loss.backward()

            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # 更新模型参数
            optimizer.step()

            total_loss += loss.item()
            total_labeled_loss += labeled_loss.item()
            total_pseudo_loss += pseudo_loss.item()

        # 打印当前epoch的平均损失
        avg_loss = total_loss / len(train_dataloader)
        avg_labeled_loss = total_labeled_loss / len(train_dataloader)
        avg_pseudo_loss = total_pseudo_loss / len(train_dataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_loss:.4f}, '
              f'Labeled Loss: {avg_labeled_loss:.4f}, Pseudo Loss: {avg_pseudo_loss:.4f}, '
              f'Pseudo Labels: {num_pseudo_labels}')

        # 每eval_interval个epoch评估一次
        if (epoch + 1) % eval_interval == 0:
            accuracy, precision, recall, f1, auc = evaluate(model, test_dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%, '
                  f'Precision: {precision:.2f}%, Recall: {recall:.2f}, '
                  f'F1 Score: {f1:.2f}, AUC: {auc:.2f}')

            # 保存训练历史
            train_history['loss'].append(avg_loss)
            train_history['accuracy'].append(accuracy)
            train_history['precision'].append(precision)
            train_history['recall'].append(recall)
            train_history['f1'].append(f1)
            train_history['auc'].append(auc)

            # 保存模型
            save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_history': train_history
            }, save_path)
            print(f'Model saved to {save_path}')


# 初始化并训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROEN(node_in_channels=1,
             edge_in_channels=55,
             hidden_channels_node=128,
             hidden_channels_edge=128,
             mlp_hidden_channels=128,
             num_edge_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_dataloader, test_dataloader, optimizer, criterion, 150, 10, 'models/2012')