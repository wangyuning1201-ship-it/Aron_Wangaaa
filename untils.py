# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
#
#
# # 创建IP到节点的动态映射，每个时间窗口更新一次
# def create_ip_mapping_2017(time_slice):
#     ip_list = pd.concat([time_slice[' Source IP'], time_slice[' Destination IP']]).unique()
#     return {ip: i for i, ip in enumerate(ip_list)}
#
#
# def create_ip_mapping_2012(time_slice):
#     ip_list = pd.concat([time_slice['source'], time_slice['destination']]).unique()
#     return {ip: i for i, ip in enumerate(ip_list)}
#
#
# def create_ip_mapping_2020(time_slice):
#     ip_list = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
#     return {ip: i for i, ip in enumerate(ip_list)}
#
#
# def create_graph_data_2017(time_slice, ip_to_id, time_window, labeled_ratio=0.2):
#     """
#     创建图数据，支持半监督学习
#     labeled_ratio: 有标签边的比例，其余边将标记为未标注
#     """
#     label_encoder = LabelEncoder()
#
#     # 将Inf替换为NaN以便插值
#     time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)
#
#     # 向前和向后填充NaN值
#     time_slice.fillna(method='ffill', inplace=True)
#     time_slice.fillna(method='bfill', inplace=True)
#
#     # 选择数值列（严格排除非数值和字符串类型）
#     numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(
#         ['Flow ID', ' Source IP', ' Destination IP', ' Label'])
#
#     # 只对数值列进行标准化
#     scaler = StandardScaler()
#     time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
#
#     # 将源IP和目标IP转换为节点ID
#     source_ids = time_slice[' Source IP'].map(ip_to_id).values
#     destination_ids = time_slice[' Destination IP'].map(ip_to_id).values
#
#     # 构建边索引（源->目标）
#     edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)
#
#     # 获取源端口和目标端口作为节点特征
#     source_ports = time_slice[' Source Port'].values
#     destination_ports = time_slice[' Destination Port'].values
#
#     # 构建节点特征矩阵
#     num_nodes = len(ip_to_id)  # 节点总数
#     node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # 初始化节点特征
#
#     # 将每个节点的特征填充到对应位置
#     for i, (src_id, dst_id, src_port, dst_port) in enumerate(
#             zip(source_ids, destination_ids, source_ports, destination_ports)):
#         node_features[src_id] = src_port  # 源IP节点特征是源端口
#         node_features[dst_id] = dst_port  # 目标IP节点特征是目标端口
#
#     # 边特征：删除'Source IP', 'Destination IP', 'Label', 'Flow ID', ' Timestamp'，使用其他数值列作为边特征
#     edge_attr_df = time_slice.drop(columns=[' Source IP', ' Destination IP', ' Label', 'Flow ID', ' Timestamp'])
#
#     # 转换为张量
#     edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
#
#     # 使用LabelEncoder将'Label'列编码为整数
#     labels_encoded = label_encoder.fit_transform(time_slice[' Label'])
#
#     # 只有存在边时才处理标签
#     if edge_index.size(1) > 0:
#         edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
#
#         # 半监督学习：创建标签掩码，只标记部分边
#         num_edges = len(edge_labels)
#         num_labeled = int(num_edges * labeled_ratio)
#
#         # 随机选择要标记的边
#         labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#         labeled_indices = torch.randperm(num_edges)[:num_labeled]
#         labeled_mask[labeled_indices] = True
#
#         # 创建未标记边的标签（设置为-1表示未标记）
#         semi_supervised_labels = edge_labels.clone()
#         semi_supervised_labels[~labeled_mask] = -1  # -1表示未标记边
#
#         graph_data = Data(
#             x=node_features,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             edge_labels=semi_supervised_labels,
#             labeled_mask=labeled_mask  # 存储哪些边有标签
#         )
#     else:
#         print(f"跳过时间窗口 {time_window}，因为没有边")
#         # 即使没有边，也创建一个空的图数据对象，但确保包含必要的属性
#         graph_data = Data(
#             x=torch.zeros((0, 1), dtype=torch.float),
#             edge_index=torch.zeros((2, 0), dtype=torch.long),
#             edge_attr=torch.zeros((0, edge_attr_df.shape[1]), dtype=torch.float),
#             edge_labels=torch.full((0,), -1, dtype=torch.long),
#             labeled_mask=torch.zeros(0, dtype=torch.bool)
#         )
#
#     # 确保图数据包含所有必要的属性
#     if not hasattr(graph_data, 'edge_labels'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.edge_labels = torch.full((num_edges,), -1, dtype=torch.long)
#
#     if not hasattr(graph_data, 'labeled_mask'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#
#     return graph_data
#
#
# def create_graph_data_2012(time_slice, ip_to_id, time_window, labeled_ratio=0.2):
#     """创建图数据，支持半监督学习"""
#     label_encoder = LabelEncoder()
#
#     # 将Inf替换为NaN以便插值
#     time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)
#
#     # 向前和向后填充NaN值
#     time_slice.fillna(method='ffill', inplace=True)
#     time_slice.fillna(method='bfill', inplace=True)
#
#     # 选择数值列（严格排除非数值和字符串类型）
#     numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(
#         ['appName', 'source', 'destination', 'Label', 'generated'])
#
#     # 只对数值列进行标准化
#     scaler = StandardScaler()
#     time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
#
#     # 将源IP和目标IP转换为节点ID
#     source_ids = time_slice['source'].map(ip_to_id).values
#     destination_ids = time_slice['destination'].map(ip_to_id).values
#
#     # 构建边索引（源->目标）
#     edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)
#
#     # 获取源端口和目标端口作为节点特征
#     source_ports = time_slice['sourcePort'].values
#     destination_ports = time_slice['destinationPort'].values
#
#     # 构建节点特征矩阵
#     num_nodes = len(ip_to_id)  # 节点总数
#     node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # 初始化节点特征
#
#     # 将每个节点的特征填充到对应位置
#     for i, (src_id, dst_id, src_port, dst_port) in enumerate(
#             zip(source_ids, destination_ids, source_ports, destination_ports)):
#         node_features[src_id] = src_port  # 源IP节点特征是源端口
#         node_features[dst_id] = dst_port  # 目标IP节点特征是目标端口
#
#     # 边特征：删除'appName', 'source', 'destination', 'Label', 'generated'，使用其他数值列作为边特征
#     edge_attr_df = time_slice.drop(columns=['appName', 'source', 'destination', 'Label', 'generated'])
#
#     # 转换为张量
#     edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
#
#     # 使用LabelEncoder将'Label'列编码为整数
#     labels_encoded = label_encoder.fit_transform(time_slice['Label'])
#
#     # 只有存在边时才处理标签
#     if edge_index.size(1) > 0:
#         edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
#
#         # 半监督学习：创建标签掩码，只标记部分边
#         num_edges = len(edge_labels)
#         num_labeled = int(num_edges * labeled_ratio)
#
#         # 随机选择要标记的边
#         labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#         labeled_indices = torch.randperm(num_edges)[:num_labeled]
#         labeled_mask[labeled_indices] = True
#
#         # 创建未标记边的标签（设置为-1表示未标记）
#         semi_supervised_labels = edge_labels.clone()
#         semi_supervised_labels[~labeled_mask] = -1  # -1表示未标记边
#
#         graph_data = Data(
#             x=node_features,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             edge_labels=semi_supervised_labels,
#             labeled_mask=labeled_mask  # 存储哪些边有标签
#         )
#     else:
#         print(f"跳过时间窗口 {time_window}，因为没有边")
#         # 即使没有边，也创建一个空的图数据对象，但确保包含必要的属性
#         graph_data = Data(
#             x=torch.zeros((0, 1), dtype=torch.float),
#             edge_index=torch.zeros((2, 0), dtype=torch.long),
#             edge_attr=torch.zeros((0, edge_attr_df.shape[1]), dtype=torch.float),
#             edge_labels=torch.full((0,), -1, dtype=torch.long),
#             labeled_mask=torch.zeros(0, dtype=torch.bool)
#         )
#
#     # 确保图数据包含所有必要的属性
#     if not hasattr(graph_data, 'edge_labels'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.edge_labels = torch.full((num_edges,), -1, dtype=torch.long)
#
#     if not hasattr(graph_data, 'labeled_mask'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#
#     return graph_data
#
#
# def create_graph_data_2020(time_slice, ip_to_id, time_window, labeled_ratio=0.2):
#     """创建图数据，支持半监督学习"""
#     label_encoder = LabelEncoder()
#
#     # 将Inf替换为NaN以便插值
#     time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)
#
#     # 向前和向后填充NaN值
#     time_slice.fillna(method='ffill', inplace=True)
#     time_slice.fillna(method='bfill', inplace=True)
#
#     # 选择数值列（严格排除非数值和字符串类型）
#     numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(
#         ['Src IP', 'Dst IP', 'Flow ID', 'Label'])
#
#     # 只对数值列进行标准化
#     scaler = StandardScaler()
#     time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
#
#     # 将源IP和目标IP转换为节点ID
#     source_ids = time_slice['Src IP'].map(ip_to_id).values
#     destination_ids = time_slice['Dst IP'].map(ip_to_id).values
#
#     # 构建边索引（源->目标）
#     edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)
#
#     # 获取源端口和目标端口作为节点特征
#     source_ports = time_slice['Src Port'].values
#     destination_ports = time_slice['Dst Port'].values
#
#     # 构建节点特征矩阵
#     num_nodes = len(ip_to_id)  # 节点总数
#     node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # 初始化节点特征
#
#     # 将每个节点的特征填充到对应位置
#     for i, (src_id, dst_id, src_port, dst_port) in enumerate(
#             zip(source_ids, destination_ids, source_ports, destination_ports)):
#         node_features[src_id] = src_port  # 源IP节点特征是源端口
#         node_features[dst_id] = dst_port  # 目标IP节点特征是目标端口
#
#     # 边特征：删除'Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp'，使用其他数值列作为边特征
#     edge_attr_df = time_slice.drop(columns=['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp'])
#
#     # 转换为张量
#     edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
#
#     # 使用LabelEncoder将'Label'列编码为整数
#     labels_encoded = label_encoder.fit_transform(time_slice['Label'])
#
#     # 只有存在边时才处理标签
#     if edge_index.size(1) > 0:
#         edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
#
#         # 半监督学习：创建标签掩码，只标记部分边
#         num_edges = len(edge_labels)
#         num_labeled = int(num_edges * labeled_ratio)
#
#         # 随机选择要标记的边
#         labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#         labeled_indices = torch.randperm(num_edges)[:num_labeled]
#         labeled_mask[labeled_indices] = True
#
#         # 创建未标记边的标签（设置为-1表示未标记）
#         semi_supervised_labels = edge_labels.clone()
#         semi_supervised_labels[~labeled_mask] = -1  # -1表示未标记边
#
#         graph_data = Data(
#             x=node_features,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             edge_labels=semi_supervised_labels,
#             labeled_mask=labeled_mask  # 存储哪些边有标签
#         )
#     else:
#         print(f"跳过时间窗口 {time_window}，因为没有边")
#         # 即使没有边，也创建一个空的图数据对象，但确保包含必要的属性
#         graph_data = Data(
#             x=torch.zeros((0, 1), dtype=torch.float),
#             edge_index=torch.zeros((2, 0), dtype=torch.long),
#             edge_attr=torch.zeros((0, edge_attr_df.shape[1]), dtype=torch.float),
#             edge_labels=torch.full((0,), -1, dtype=torch.long),
#             labeled_mask=torch.zeros(0, dtype=torch.bool)
#         )
#
#     # 确保图数据包含所有必要的属性
#     if not hasattr(graph_data, 'edge_labels'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.edge_labels = torch.full((num_edges,), -1, dtype=torch.long)
#
#     if not hasattr(graph_data, 'labeled_mask'):
#         num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
#         graph_data.labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
#
#     return graph_data
#
#
# # 自定义数据集
# class GraphDataset(Dataset):
#     def __init__(self, graph_data_seq, device):
#         self.graph_data_seq = graph_data_seq
#         self.device = device
#
#     def __len__(self):
#         return len(self.graph_data_seq)
#
#     def __getitem__(self, idx):
#         graph_data = self.graph_data_seq[idx]
#
#         # 确保每个graph_data的数据都移动到指定设备
#         graph_data.x = graph_data.x.to(self.device)
#         graph_data.edge_index = graph_data.edge_index.to(self.device)
#         graph_data.edge_attr = graph_data.edge_attr.to(self.device)
#         graph_data.edge_labels = graph_data.edge_labels.to(self.device)
#         graph_data.labeled_mask = graph_data.labeled_mask.to(self.device)
#
#         return graph_data
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
from datetime import datetime, timedelta


# 创建IP到节点的动态映射
def create_ip_mapping(time_slice):
    # 使用提供的列名
    ip_columns = [' Source IP', ' Destination IP']

    ip_list = pd.concat([time_slice[ip_columns[0]], time_slice[ip_columns[1]]]).unique()
    return {ip: i for i, ip in enumerate(ip_list)}


# 创建图数据函数，处理完整的数据集
def create_graph_data(time_slice, ip_to_id, time_window, labeled_ratio=0.2):
    """创建图数据，支持半监督学习"""
    label_encoder = LabelEncoder()

    # 将Inf替换为NaN以便插值
    time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 向前和向后填充NaN值
    time_slice.fillna(method='ffill', inplace=True)
    time_slice.fillna(method='bfill', inplace=True)

    # 选择数值列（排除非数值和字符串类型）
    exclude_cols = ['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port',
                    ' Protocol', ' Timestamp', ' Label']

    numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols)

    # 只对数值列进行标准化
    scaler = StandardScaler()
    time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])

    # 将源IP和目标IP转换为节点ID
    source_ids = time_slice[' Source IP'].map(ip_to_id).values
    destination_ids = time_slice[' Destination IP'].map(ip_to_id).values

    # 构建边索引（源->目标）
    edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)

    # 获取源端口和目标端口作为节点特征
    source_ports = time_slice[' Source Port'].values
    destination_ports = time_slice[' Destination Port'].values

    # 构建节点特征矩阵
    num_nodes = len(ip_to_id)  # 节点总数
    node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # 初始化节点特征

    # 将每个节点的特征填充到对应位置
    for i, (src_id, dst_id, src_port, dst_port) in enumerate(
            zip(source_ids, destination_ids, source_ports, destination_ports)):
        node_features[src_id] = src_port  # 源IP节点特征是源端口
        node_features[dst_id] = dst_port  # 目标IP节点特征是目标端口

    # 边特征：排除非数值列
    edge_attr_df = time_slice.drop(columns=exclude_cols)

    # 转换为张量
    edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)

    # 使用LabelEncoder将标签列编码为整数
    labels_encoded = label_encoder.fit_transform(time_slice[' Label'])

    # 只有存在边时才处理标签
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels_encoded, dtype=torch.long)

        # 半监督学习：创建标签掩码，只标记部分边
        num_edges = len(edge_labels)
        num_labeled = int(num_edges * labeled_ratio)

        # 随机选择要标记的边
        labeled_mask = torch.zeros(num_edges, dtype=torch.bool)
        labeled_indices = torch.randperm(num_edges)[:num_labeled]
        labeled_mask[labeled_indices] = True

        # 创建未标记边的标签（设置为-1表示未标记）
        semi_supervised_labels = edge_labels.clone()
        semi_supervised_labels[~labeled_mask] = -1  # -1表示未标记边

        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_labels=semi_supervised_labels,
            labeled_mask=labeled_mask  # 存储哪些边有标签
        )
    else:
        print(f"跳过时间窗口 {time_window}，因为没有边")
        # 即使没有边，也创建一个空的图数据对象，但确保包含必要的属性
        graph_data = Data(
            x=torch.zeros((0, 1), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, edge_attr_df.shape[1]), dtype=torch.float),
            edge_labels=torch.full((0,), -1, dtype=torch.long),
            labeled_mask=torch.zeros(0, dtype=torch.bool)
        )

    # 确保图数据包含所有必要的属性
    if not hasattr(graph_data, 'edge_labels'):
        num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
        graph_data.edge_labels = torch.full((num_edges,), -1, dtype=torch.long)

    if not hasattr(graph_data, 'labeled_mask'):
        num_edges = graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') else 0
        graph_data.labeled_mask = torch.zeros(num_edges, dtype=torch.bool)

    return graph_data


# 自定义数据集
class GraphDataset(Dataset):
    def __init__(self, graph_data_seq, device):
        self.graph_data_seq = graph_data_seq
        self.device = device

    def __len__(self):
        return len(self.graph_data_seq)

    def __getitem__(self, idx):
        graph_data = self.graph_data_seq[idx]

        # 确保每个graph_data的数据都移动到指定设备
        graph_data.x = graph_data.x.to(self.device)
        graph_data.edge_index = graph_data.edge_index.to(self.device)
        graph_data.edge_attr = graph_data.edge_attr.to(self.device)
        graph_data.edge_labels = graph_data.edge_labels.to(self.device)
        graph_data.labeled_mask = graph_data.labeled_mask.to(self.device)

        return graph_data


# 主函数，处理完整数据集并按时间窗口分割
def process_dataset(file_path, time_window_minutes=1, labeled_ratio=0.2):
    """处理完整数据集并按时间窗口分割"""

    # 创建保存目录
    save_dir = "graph_data"
    os.makedirs(save_dir, exist_ok=True)
    print(f"图数据将保存到目录: {save_dir}")

    # 读取数据
    print(f"正在读取数据: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"成功读取数据，形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None

    # 转换时间戳列
    try:
        data[' Timestamp'] = pd.to_datetime(data[' Timestamp'])
        print(f"时间戳转换成功，时间范围: {data[' Timestamp'].min()} 到 {data[' Timestamp'].max()}")
    except Exception as e:
        print(f"时间戳转换失败: {e}")
        return None

    # 按时间窗口分组
    start_time = data[' Timestamp'].min()
    end_time = data[' Timestamp'].max()

    # 创建时间窗口
    time_windows = []
    current_time = start_time
    while current_time <= end_time:
        next_time = current_time + timedelta(minutes=time_window_minutes)
        time_windows.append((current_time, next_time))
        current_time = next_time

    print(f"创建了 {len(time_windows)} 个时间窗口")

    # 处理每个时间窗口
    graph_data_list = []

    for i, (window_start, window_end) in enumerate(time_windows):
        # 筛选当前时间窗口的数据
        window_data = data[(data[' Timestamp'] >= window_start) & (data[' Timestamp'] < window_end)]

        if len(window_data) == 0:
            print(f"时间窗口 {window_start} 到 {window_end} 没有数据，跳过")
            continue

        print(f"处理时间窗口 {i + 1}/{len(time_windows)}: {window_start} 到 {window_end}, 数据量: {len(window_data)}")

        # 创建IP映射
        ip_to_id = create_ip_mapping(window_data)

        # 创建图数据
        graph_data = create_graph_data(window_data, ip_to_id, f"{window_start}_{window_end}", labeled_ratio)

        # 保存图数据到graph_data文件夹
        time_str = window_start.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(save_dir, f"graph_{time_str}.pt")
        torch.save(graph_data, save_path)
        print(f"图数据已保存到: {save_path}")

        graph_data_list.append(graph_data)

    return graph_data_list


# 使用示例
if __name__ == "__main__":
    file_path = r"C:\Users\Aron\Desktop\Tuesday-WorkingHours.pcap_ISCX.csv"
    graph_data_list = process_dataset(file_path, time_window_minutes=1)

    if graph_data_list:
        print(f"成功处理了 {len(graph_data_list)} 个时间窗口的图数据")

        # 统计信息
        total_nodes = sum([g.x.shape[0] for g in graph_data_list])
        total_edges = sum([g.edge_index.shape[1] for g in graph_data_list])
        total_labeled_edges = sum([torch.sum(g.labeled_mask).item() for g in graph_data_list])

        print(f"总计:")
        print(f"  节点数: {total_nodes}")
        print(f"  边数: {total_edges}")
        print(f"  有标签的边数: {total_labeled_edges}")

        # 创建数据集
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = GraphDataset(graph_data_list, device)
        print(f"数据集创建成功，包含 {len(dataset)} 个图")