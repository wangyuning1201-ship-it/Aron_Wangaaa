import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, accuracy_score
from untils import create_ip_mapping, create_graph_data, GraphDataset
from network import ROEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch_geometric.loader import DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
from torch_geometric.data.data import DataEdgeAttr

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加安全全局类以支持 PyTorch Geometric 数据加载
torch.serialization.add_safe_globals([DataEdgeAttr])


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# 直接从保存的图数据加载
graph_data_dir = r"D:\WYNproject\semi-roen\graph_data"

# 加载所有图数据文件（排除 metadata.pt）
graph_data_files = [f for f in glob.glob(os.path.join(graph_data_dir, "*.pt"))
                    if not f.endswith("metadata.pt")]
graph_data_seq = []

for file_path in graph_data_files:
    try:
        # 使用 weights_only=False 加载图数据
        graph_data = torch.load(file_path, weights_only=False)
        graph_data_seq.append(graph_data)
        print(f"成功加载: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}")

# 检查是否成功加载了图数据
if not graph_data_seq:
    raise ValueError("未加载任何图数据。请检查目录路径和文件格式。")

print(f"总共加载了 {len(graph_data_seq)} 个图数据")

# 初始化设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 检查数据样本数量并适当调整数据划分
n_samples = len(graph_data_seq)
if n_samples <= 1:
    # 样本太少，无法划分
    train_data_seq = graph_data_seq
    test_data_seq = []
    print("警告: 样本太少，无法划分训练集和测试集")
elif n_samples <= 5:
    # 样本很少，使用留一法
    test_size = 1
    train_data_seq, test_data_seq = train_test_split(
        graph_data_seq, test_size=test_size, random_state=42, shuffle=True
    )
else:
    # 样本足够，使用正常划分
    train_data_seq, test_data_seq = train_test_split(
        graph_data_seq, test_size=0.2, random_state=42, shuffle=True
    )

print(f"训练集大小: {len(train_data_seq)}, 测试集大小: {len(test_data_seq)}")

# 为训练集和测试集创建DataLoader
train_dataset = GraphDataset(train_data_seq, device=device)
test_dataset = GraphDataset(test_data_seq, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=min(32, len(train_data_seq)), shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=min(32, len(test_data_seq)), shuffle=False, num_workers=0)


def evaluate(model, dataloader):
    """评估模型性能，考虑半监督学习场景"""
    model.eval()  # 将模型设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    all_labels = []
    all_preds = []
    all_labeled_mask = []  # 存储哪些边有标签
    all_probs = []  # 存储预测概率

    with torch.no_grad():  # 关闭梯度计算
        for batch in dataloader:
            batch = batch.to(device)

            # 获取模型预测
            edge_predictions, _ = model(graphs=[batch], seq_len=1, train_mode=False)
            edge_labels_batch = batch.edge_labels.to(device)
            labeled_mask = batch.labeled_mask.to(device)  # 获取标签掩码

            # 获取每条边的预测类别
            probs = torch.softmax(edge_predictions[0], dim=1)
            _, predicted = torch.max(probs, dim=1)

            # 存储真实标签和预测结果
            all_labels.extend(edge_labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labeled_mask.extend(labeled_mask.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_labeled_mask = np.array(all_labeled_mask)
    all_probs = np.array(all_probs)

    # 只计算有标签边的指标
    labeled_indices = np.where(all_labeled_mask == 1)[0]

    if len(labeled_indices) == 0:
        print("警告: 没有找到有标签的边")
        return 0, 0, 0, 0, 0, all_labels, all_preds, all_probs

    labeled_labels = all_labels[labeled_indices]
    labeled_preds = all_preds[labeled_indices]
    labeled_probs = all_probs[labeled_indices]

    # 计算准确率
    accuracy = 100 * accuracy_score(labeled_labels, labeled_preds)

    # 计算召回率
    recall = recall_score(labeled_labels, labeled_preds, average='weighted', zero_division=0)

    # 计算F1分数
    f1 = f1_score(labeled_labels, labeled_preds, average='weighted', zero_division=0)

    # 计算精确率
    precision = 100 * precision_score(labeled_labels, labeled_preds, average='weighted', zero_division=0)

    # 计算AUC值
    try:
        # 对于二分类问题
        if len(np.unique(labeled_labels)) == 2:
            auc = roc_auc_score(labeled_labels, labeled_probs[:, 1])
        else:
            # 对于多分类，使用One-vs-Rest策略
            auc = roc_auc_score(labeled_labels, labeled_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')  # 如果无法计算AUC，返回NaN

    model.train()  # 将模型恢复为训练模式

    # 返回评估指标和详细结果
    return accuracy, precision, recall, f1, auc, labeled_labels, labeled_preds, labeled_probs


def train(model, train_dataloader, test_dataloader, optimizer, criterion, num_epochs, eval_interval, save_dir):
    """训练模型，使用半监督学习"""
    model.train()  # 将模型设置为训练模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 存储训练历史
    train_history = {
        'epoch': [],
        'train_loss': [],
        'labeled_loss': [],
        'pseudo_loss': [],
        'pseudo_labels_count': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }

    # 如果没有测试数据，只训练不评估
    no_test_data = len(test_dataloader) == 0

    # 用于检测过拟合的变量
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # 早停耐心值

    for epoch in range(num_epochs):
        total_loss = 0
        total_labeled_loss = 0
        total_pseudo_loss = 0
        num_pseudo_labels = 0
        num_batches = 0

        # 训练阶段
        for batch in train_dataloader:  # 遍历训练数据
            batch = batch.to(device)

            optimizer.zero_grad()

            # 获取预测结果和伪标签
            edge_predictions, pseudo_labels_list = model(graphs=[batch], seq_len=1, train_mode=True)

            # 获取当前批次的边标签
            edge_labels_batch = batch.edge_labels.to(device)
            labeled_mask = batch.labeled_mask.to(device)  # 获取标签掩码

            # 计算有标签边的损失
            if torch.sum(labeled_mask) > 0:  # 确保有标签的边
                labeled_loss = criterion(edge_predictions[0][labeled_mask], edge_labels_batch[labeled_mask])
            else:
                labeled_loss = torch.tensor(0.0, device=device)

            # 计算伪标签损失（如果有）
            pseudo_loss = torch.tensor(0.0, device=device)
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
            num_batches += 1

        # 计算平均损失
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_labeled_loss = total_labeled_loss / num_batches
            avg_pseudo_loss = total_pseudo_loss / num_batches
        else:
            avg_loss = avg_labeled_loss = avg_pseudo_loss = 0

        # 记录训练指标
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_loss)
        train_history['labeled_loss'].append(avg_labeled_loss)
        train_history['pseudo_loss'].append(avg_pseudo_loss)
        train_history['pseudo_labels_count'].append(num_pseudo_labels)

        print(f'轮次 {epoch + 1}/{num_epochs}, 总损失: {avg_loss:.4f}, '
              f'有标签损失: {avg_labeled_loss:.4f}, 伪标签损失: {avg_pseudo_loss:.4f}, '
              f'伪标签数量: {num_pseudo_labels}')

        # 每eval_interval个epoch评估一次（如果有测试数据）
        if (epoch + 1) % eval_interval == 0 and not no_test_data:
            accuracy, precision, recall, f1, auc, _, _, _ = evaluate(model, test_dataloader)
            print(f'轮次 {epoch + 1}/{num_epochs}, 测试准确率: {accuracy:.2f}%, '
                  f'精确率: {precision:.2f}%, 召回率: {recall:.2f}, '
                  f'F1分数: {f1:.2f}, AUC: {auc:.2f}')

            # 记录验证指标
            train_history['val_accuracy'].append(accuracy)
            train_history['val_precision'].append(precision)
            train_history['val_recall'].append(recall)
            train_history['val_f1'].append(f1)
            train_history['val_auc'].append(auc)

            # 早停检测
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'train_history': train_history
                }, save_path)
                print(f'最佳模型已保存至 {save_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'在第 {epoch + 1} 轮次触发早停')
                    break

        # 定期保存检查点
        if (epoch + 1) % (eval_interval * 5) == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_history': train_history
            }, save_path)
            print(f'检查点已保存至 {save_path}')

    # 训练结束后保存最终模型
    save_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'train_history': train_history
    }, save_path)
    print(f'最终模型已保存至 {save_path}')

    return train_history


def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='总损失')
    plt.plot(history['epoch'], history['labeled_loss'], label='有标签损失')
    plt.plot(history['epoch'], history['pseudo_loss'], label='伪标签损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    # 绘制伪标签数量曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['pseudo_labels_count'], 'g-')
    plt.xlabel('训练轮次')
    plt.ylabel('伪标签数量')
    plt.title('使用的伪标签数量')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

    # 如果有验证指标，绘制验证指标曲线
    if history['val_accuracy']:
        eval_epochs = history['epoch'][::10]  # 每10个epoch评估一次

        plt.figure(figsize=(15, 10))

        # 准确率
        plt.subplot(2, 3, 1)
        plt.plot(eval_epochs, history['val_accuracy'], 'b-')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率 (%)')
        plt.title('验证准确率')
        plt.grid(True)

        # 精确率
        plt.subplot(2, 3, 2)
        plt.plot(eval_epochs, history['val_precision'], 'r-')
        plt.xlabel('训练轮次')
        plt.ylabel('精确率 (%)')
        plt.title('验证精确率')
        plt.grid(True)

        # 召回率
        plt.subplot(2, 3, 3)
        plt.plot(eval_epochs, history['val_recall'], 'g-')
        plt.xlabel('训练轮次')
        plt.ylabel('召回率')
        plt.title('验证召回率')
        plt.grid(True)

        # F1分数
        plt.subplot(2, 3, 4)
        plt.plot(eval_epochs, history['val_f1'], 'm-')
        plt.xlabel('训练轮次')
        plt.ylabel('F1分数')
        plt.title('验证F1分数')
        plt.grid(True)

        # AUC
        plt.subplot(2, 3, 5)
        plt.plot(eval_epochs, history['val_auc'], 'c-')
        plt.xlabel('训练轮次')
        plt.ylabel('AUC')
        plt.title('验证AUC')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_metrics.png'))
        plt.close()

        # 检测过拟合：比较训练损失和验证性能
        plt.figure(figsize=(10, 5))

        # 归一化损失和准确率以便在同一尺度上比较
        norm_train_loss = np.array(history['train_loss']) / max(history['train_loss'])
        norm_val_accuracy = np.array(history['val_accuracy']) / max(history['val_accuracy'])

        plt.plot(eval_epochs, norm_train_loss[::10], 'b-', label='归一化训练损失')
        plt.plot(eval_epochs, norm_val_accuracy, 'r-', label='归一化验证准确率')
        plt.xlabel('训练轮次')
        plt.ylabel('归一化值')
        plt.title('过拟合检测: 训练损失 vs 验证准确率')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overfitting_detection.png'))
        plt.close()


def print_final_results(model, test_dataloader, history, save_dir):
    """打印最终结果并分析模型性能"""
    print("\n" + "=" * 50)
    print("最终结果")
    print("=" * 50)

    # 评估模型
    if test_dataloader:
        accuracy, precision, recall, f1, auc, labels, preds, probs = evaluate(model, test_dataloader)

        print(f"测试准确率: {accuracy:.2f}%")
        print(f"测试精确率: {precision:.2f}%")
        print(f"测试召回率: {recall:.4f}")
        print(f"测试F1分数: {f1:.4f}")
        print(f"测试AUC: {auc:.4f}")

        # 保存详细结果
        results_df = pd.DataFrame({
            '真实标签': labels,
            '预测标签': preds,
            '类别0概率': probs[:, 0],
            '类别1概率': probs[:, 1] if probs.shape[1] > 1 else [0] * len(labels)
        })
        results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        print("\n混淆矩阵:")
        print(cm)

        # 保存混淆矩阵
        np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
    else:
        print("没有测试数据可用于最终评估")

    # 分析半监督学习效果
    print("\n" + "=" * 50)
    print("半监督学习分析")
    print("=" * 50)

    pseudo_counts = history['pseudo_labels_count']
    print(f"每轮次平均伪标签数量: {np.mean(pseudo_counts):.2f}")
    print(f"单轮次最大伪标签数量: {max(pseudo_counts)}")
    print(f"训练过程中使用的总伪标签数量: {sum(pseudo_counts)}")

    if len(pseudo_counts) > 0 and max(pseudo_counts) > 0:
        print("✓ 模型成功使用了伪标签技术（半监督学习）")
    else:
        print("✗ 模型未能有效使用伪标签技术")

    # 分析过拟合
    print("\n" + "=" * 50)
    print("过拟合分析")
    print("=" * 50)

    if history['val_accuracy']:
        # 检查验证性能是否在后期下降
        last_quarter = len(history['val_accuracy']) // 4
        early_acc = np.mean(history['val_accuracy'][:last_quarter])
        late_acc = np.mean(history['val_accuracy'][-last_quarter:])

        if late_acc < early_acc - 5:  # 允许5%的波动
            print("✓ 检测到过拟合迹象: 验证性能下降")
        else:
            print("✗ 未检测到明显的过拟合迹象")

        # 检查训练损失和验证性能的相关性
        corr = np.corrcoef(history['train_loss'][::10], history['val_accuracy'])[0, 1]
        print(f"训练损失与验证准确率的相关性: {corr:.4f}")
    else:
        print("没有验证数据可用于过拟合分析")

    # 绘制训练历史
    plot_training_history(history, save_dir)
    print(f"\n训练图表已保存至 {save_dir}")


# 初始化并训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROEN(node_in_channels=1,
             edge_in_channels=77,
             hidden_channels_node=128,
             hidden_channels_edge=128,
             mlp_hidden_channels=128,
             num_edge_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
save_dir = 'models/2012'
history = train(model, train_dataloader, test_dataloader, optimizer, criterion, 150, 10, save_dir)

# 打印最终结果
print_final_results(model, test_dataloader, history, save_dir)