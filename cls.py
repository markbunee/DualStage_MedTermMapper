import os
import torch
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import pickle
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
#######################################################
# 参数配置
config = {
    'data_path': r'F:\final\dataset\train_pair_dataset.csv',
    'local_model_path': r'F:\final\premodel',
    'checkpoint_dir': './checkpoint_cls',  # 修改为分类任务的保存路径
    'max_seq_length': 100,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 3e-5,
    'scheduler_start_factor': 1.0,
    'scheduler_end_factor': 0.01,
    'k_folds': 10,
    'random_seed': 42,
    'log_dir': './logs_cls', 
}

# 数据加载与预处理
train_data_df = pd.read_csv(
    config['data_path'], 
    sep=',', 
    header=0,  # 假设第一行是列名
    names=['text_a', 'text_b', 'label'],  # 如果文件没有列名，则使用此参数指定列名
    engine='python'
)

# 确保标签是整数类型
try:
    train_data_df['label'] = train_data_df['label'].astype(int)  # 将标签转换为整数类型
except ValueError as e:
    print(f"Error converting label to int: {e}")
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
        encoding = self.tokenizer(
            text_a,
            text_b,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # 标签改为整数类型
        }

# 加载本地模型和分词器
tokenizer = AutoTokenizer.from_pretrained(config['local_model_path'])

# 创建自定义数据集
train_dataset = CustomDataset(train_data_df, tokenizer, config['max_seq_length'])

# 使用 K 折交叉验证
kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=config['random_seed'])
# 确保日志目录存在
os.makedirs(config['log_dir'], exist_ok=True)

for fold_, (train_ids, dev_ids) in enumerate(kf.split(train_dataset)):
    print(f"Training fold {fold_}")
    
    # 划分训练集和验证集
    train_fold_dataset = torch.utils.data.Subset(train_dataset, train_ids)
    dev_fold_dataset = torch.utils.data.Subset(train_dataset, dev_ids)

    # 创建 DataLoader
    train_loader = DataLoader(train_fold_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_fold_dataset, batch_size=config['batch_size'], shuffle=False)

    # 加载分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        config['local_model_path'],
        num_labels=len(train_data_df['label'].unique())  # 根据标签类别数设置 num_labels
    ).cuda()

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
    writer = SummaryWriter(log_dir=f'runs_cls/fold_{fold_}')  # TensorBoard 日志目录

    # 确保日志目录存在
    os.makedirs(config['log_dir'], exist_ok=True)

    # 指定日志文件路径
    log_file_path = os.path.join(config['log_dir'], f'log_fold_{fold_}.txt')
    log_file = open(log_file_path, 'w')  # 打开日志文件

    # 训练模型
    model.train()
    for epoch in range(config['num_epochs']):  # 训练多个 epoch
        print(f"Epoch {epoch + 1} / {config['num_epochs']}")
        train_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 反向传播和优化
            loss.backward()
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 收集预测结果和真实标签
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # 累加验证损失
            val_loss += loss.item()

    # 计算验证集的指标
    avg_val_loss = val_loss / len(dev_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # 记录验证损失和指标到 TensorBoard 和日志文件
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('F1/val', f1, epoch)
    log_file.write(f"Fold {fold_}, Epoch {epoch + 1}, Val Loss: {avg_val_loss}, Accuracy: {accuracy}, F1: {f1}\n")

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