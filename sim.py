import os
import torch
import pandas as pd
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# 参数配置
config = {
    'data_path': r'F:\final\dataset\train_pair_dataset.csv',
    'local_model_path': r'F:\final\premodel',
    'checkpoint_dir': './checkpoint_sim',
    'max_seq_length': 100,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 3e-5,
    'scheduler_start_factor': 1.0,
    'scheduler_end_factor': 0.01,
    'k_folds': 10,
    'random_seed': 42,
    'log_dir': './logs_sim', 
}

# 数据加载与预处理
train_data_df = pd.read_csv(
    config['data_path'], 
    sep=',', 
    header=0,  # 假设第一行是列名
    names=['text_a', 'text_b', 'label'],  # 如果文件没有列名，则使用此参数指定列名
    engine='python'
)

# 确保标签是数值类型
try:
    train_data_df['label'] = train_data_df['label'].astype(float)  # 尝试转换为 float 类型
except ValueError as e:
    print(f"Error converting label to float: {e}")
    # 进一步处理或修正数据

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_a = self.data.iloc[idx]['text_a']
        text_b = self.data.iloc[idx]['text_b']
        label = self.data.iloc[idx]['label']
        
        # Tokenize 输入文本
        encoding_a = self.tokenizer(
            text_a,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding_b = self.tokenizer(
            text_b,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': encoding_a['input_ids'].squeeze(0),
            'attention_mask_a': encoding_a['attention_mask'].squeeze(0),
            'input_ids_b': encoding_b['input_ids'].squeeze(0),
            'attention_mask_b': encoding_b['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)  # 标签改为 float 类型
        }

# 加载本地模型和分词器
tokenizer = BertTokenizer.from_pretrained(config['local_model_path'])

# 创建自定义数据集
train_dataset = CustomDataset(train_data_df, tokenizer, config['max_seq_length'])

# 使用 K 折交叉验证
kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=config['random_seed'])

# 确保日志目录存在
os.makedirs(config['log_dir'], exist_ok=True)
################################0312
# 定义函数：动态寻找最佳阈值
def find_best_threshold(similarities, true_labels):
    """
    动态寻找最佳相似度阈值，最大化 F1 分数。
    :param similarities: 模型预测的相似度分数 (numpy array)
    :param true_labels: 真实标签 (numpy array)
    :return: 最佳阈值和对应的 F1 分数
    """
    best_threshold = 0.5  # 默认初始阈值
    best_f1 = 0.0         # 初始最佳 F1 分数

    # 遍历候选阈值
    for threshold in np.arange(0.0, 1.01, 0.01):  # 步长为 0.01
        predicted_labels = (similarities > threshold).astype(int)  # 根据阈值生成预测标签
        f1 = f1_score(true_labels, predicted_labels, average='binary')  # 计算 F1 分数

        # 更新最佳阈值和 F1 分数
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

for fold_, (train_ids, dev_ids) in enumerate(kf.split(train_dataset)):
    print(f"Training fold {fold_}")
    
    # 划分训练集和验证集
    train_fold_dataset = torch.utils.data.Subset(train_dataset, train_ids)
    dev_fold_dataset = torch.utils.data.Subset(train_dataset, dev_ids)

    # 创建 DataLoader
    train_loader = DataLoader(train_fold_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_fold_dataset, batch_size=config['batch_size'], shuffle=False)

    # 加载本地模型
    model = BertModel.from_pretrained(config['local_model_path']).cuda()

    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config['scheduler_start_factor'],
        end_factor=config['scheduler_end_factor'],
        total_iters=total_steps
    )

    # 初始化 TensorBoard writer 和 log 文件
    writer = SummaryWriter(log_dir=f'runs/fold_{fold_}')  # TensorBoard 日志目录

    # 指定日志文件路径
    log_file_path = os.path.join(config['log_dir'], f'log_fold_{fold_}.txt')  # 日志文件路径
    log_file = open(log_file_path, 'w')  # 打开日志文件

    # 训练模型
    model.train()
    for epoch in range(config['num_epochs']):  # 训练多个 epoch
        print(f"Epoch {epoch + 1} / {config['num_epochs']}")
        train_loss = 0
        for batch in tqdm(train_loader):
            input_ids_a = batch['input_ids_a'].cuda()
            attention_mask_a = batch['attention_mask_a'].cuda()
            input_ids_b = batch['input_ids_b'].cuda()
            attention_mask_b = batch['attention_mask_b'].cuda()
            labels = batch['labels'].cuda()

            # 获取句子嵌入
            outputs_a = model(input_ids=input_ids_a, attention_mask=attention_mask_a)
            outputs_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b)
            
            # 取 [CLS] 标记的向量作为句子嵌入
            sentence_embedding_a = outputs_a.last_hidden_state[:, 0, :]
            sentence_embedding_b = outputs_b.last_hidden_state[:, 0, :]

            # 计算余弦相似度
            similarity_scores = torch.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=1)

            # 将相似度分数与真实标签进行对比，计算损失
            loss_fn = torch.nn.MSELoss()  # 使用 MSE 损失函数
            loss = loss_fn(similarity_scores, labels)

            # 正常前向传播和反向传播
            loss.backward()

            # 梯度裁剪和优化器更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 累加训练损失
            train_loss += loss.item()

        # 记录训练损失到 TensorBoard 和日志文件
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        log_file.write(f"Fold {fold_}, Epoch {epoch + 1}, Train Loss: {avg_train_loss}\n")


    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        similarities = []
        true_labels = []
        for batch in dev_loader:
            input_ids_a = batch['input_ids_a'].cuda()
            attention_mask_a = batch['attention_mask_a'].cuda()
            input_ids_b = batch['input_ids_b'].cuda()
            attention_mask_b = batch['attention_mask_b'].cuda()
            labels = batch['labels'].cuda()

            # 获取句子嵌入
            outputs_a = model(input_ids=input_ids_a, attention_mask=attention_mask_a)
            outputs_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b)
            sentence_embedding_a = outputs_a.last_hidden_state[:, 0, :]
            sentence_embedding_b = outputs_b.last_hidden_state[:, 0, :]

            # 计算余弦相似度
            similarity_scores = torch.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=1)

            # 计算损失
            loss_fn = torch.nn.MSELoss()
            val_loss += loss_fn(similarity_scores, labels).item()

            # 收集相似度和真实标签用于指标计算
            similarities.extend(similarity_scores.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 转换为 numpy 数组
    similarities = np.array(similarities)
    true_labels = np.array(true_labels)

    # 计算平均验证损失
    avg_val_loss = val_loss / len(dev_loader)  # 添加此行以定义 avg_val_loss

    # 寻找最佳阈值
    best_threshold, best_f1 = find_best_threshold(similarities, true_labels)

    # 使用最佳阈值计算其他指标
    predicted_labels = (similarities > best_threshold).astype(int)
    accuracy = np.mean(predicted_labels == true_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # 记录验证损失和指标到 TensorBoard 和日志文件
    writer.add_scalar('Loss/val', avg_val_loss, epoch)  # 此处已定义 avg_val_loss
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('Precision/val', precision, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    writer.add_scalar('F1/val', best_f1, epoch)
    log_file.write(
        f"Fold {fold_}, Epoch {epoch + 1}, Val Loss: {avg_val_loss}, "
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, "
        f"F1: {best_f1}, Best Threshold: {best_threshold}\n"
    )

    # 保存模型到本地路径
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    model.save_pretrained(os.path.join(config['checkpoint_dir'], f'fold_{fold_}'))
    tokenizer.save_pretrained(os.path.join(config['checkpoint_dir'], f'fold_{fold_}'))
    print(f"Model and tokenizer saved to {os.path.join(config['checkpoint_dir'], f'fold_{fold_}')}")

    # 清理 GPU 缓存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 关闭 TensorBoard writer 和日志文件
    writer.close()
    log_file.close()