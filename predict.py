import pandas as pd
import re
import tqdm
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import json

# 数据加载
def clean_text(text):
    text = str(text)
    # 仅移除连续的数字和标点符号
    text = re.sub(r'[\d\.]+', '', text)  # 移除数字
    text = re.sub(r'[、（）ⅠⅡⅢⅣ]+', '', text)  # 移除特定标点
    text = re.sub(r'\s+', ' ', text).strip()  # 合并空白字符
    return text

# 数据加载部分添加重点词语库加载
def load_priority_terms(path):
    """加载重点词语库"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 加载数据
TEST_DATA_PATH = r"./dataset/data_test.csv"
STANDARD_TERMS_PATH = r"./dataset/map.csv"
PRIORITY_TERMS_PATH = r"./priority_terms.txt"  # #####

test_df = pd.read_csv(TEST_DATA_PATH)
standard_terms_df = pd.read_csv(STANDARD_TERMS_PATH)
priority_terms = load_priority_terms(PRIORITY_TERMS_PATH)  #####

# 数据清洗
test_df['原始诊断名称'] = test_df['原始诊断名称'].apply(clean_text)
test_df['映射结果中文同义词'] = test_df['映射结果中文同义词'].apply(clean_text)
test_df.dropna(subset=['原始诊断名称', '映射结果中文同义词'], inplace=True)

# 构建标准词库
standard_terms = standard_terms_df.set_index('映射结果中文同义词')['映射结果SCTID'].to_dict()
standard_terms_list = list(standard_terms.keys())
#############################################################################################
# 生成候选词  sim
class ClinicalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class SentenceBERT(torch.nn.Module):
    def __init__(self, model_name):
        super(SentenceBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(2)
        return pooled_output

# 初始化模型和分词器
MODEL_PATH = r"./checkpoint_sim/fold_7"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = SentenceBERT(MODEL_PATH).cuda()
model.eval()

# 创建数据集
test_dataset = ClinicalDataset(test_df['原始诊断名称'].tolist(), tokenizer, 256)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

standard_terms_dataset = ClinicalDataset(standard_terms_list, tokenizer, 256)
standard_terms_dataloader = DataLoader(standard_terms_dataset, batch_size=32, shuffle=False)

# 获取嵌入向量
def get_embeddings(dataloader):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="获取嵌入向量"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            outputs = model(input_ids, attention_mask)
            embeddings.extend(outputs.cpu().numpy())
    return np.array(embeddings)

standard_terms_embeddings = get_embeddings(standard_terms_dataloader)
test_embeddings = get_embeddings(test_dataloader)

# 提前预处理重点词语
priority_terms_set = set(priority_terms)

def get_top_k_candidates(test_embedding, standard_terms_embeddings, standard_terms_list, priority_terms_set, current_text, k=10):
    included_priority = [p for p in priority_terms_set if p in current_text]
    if included_priority:
        filtered_terms = [term for term in standard_terms_list if any(p in term for p in included_priority)]
        if filtered_terms:
            filtered_indices = [standard_terms_list.index(term) for term in filtered_terms]
            filtered_embeddings = standard_terms_embeddings[filtered_indices]
            similarities = cosine_similarity([test_embedding], filtered_embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            return [filtered_terms[i] for i in top_indices]
    similarities = cosine_similarity([test_embedding], standard_terms_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [standard_terms_list[i] for i in top_indices]

# 生成候选词时需要遍历每个测试样本
test_texts = test_df['原始诊断名称'].tolist()
test_embeddings = get_embeddings(test_dataloader)

# 新增：生成候选词需要传递更多信息
candidates = []
for i in tqdm.tqdm(range(len(test_embeddings)), desc="生成候选词"):
    embedding = test_embeddings[i]
    current_text = test_texts[i]
    candidates.append(
        get_top_k_candidates(
            embedding,
            standard_terms_embeddings,
            standard_terms_list,
            priority_terms_set,
            current_text,
            k=10
        )
    )
test_df['top_10_candidates'] = candidates

# 将候选词保存到本地文件
candidates_path = '../candidates.json'

with open(candidates_path, 'w', encoding='utf-8') as f:
    json.dump(test_df['top_10_candidates'].tolist(), f, ensure_ascii=False)
print(f"候选词已保存至 {candidates_path}")

# 重新加载候选词
with open(candidates_path, 'r', encoding='utf-8') as f:
    loaded_candidates = json.load(f)

# 确保加载的候选词长度与原始数据框一致
assert len(loaded_candidates) == len(test_df), "加载的候选词长度与原始数据框长度不一致"

# 更新 DataFrame
test_df['top_10_candidates'] = loaded_candidates

############################################################################################


### 候选词分类  cls
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CustomDataset(Dataset):
    def __init__(self, texts, candidates, tokenizer, max_length):
        self.texts = texts
        self.candidates = candidates
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
################显式遍历每个候选词，生成每个候选词对应的编码
    def __getitem__(self, idx):
        text = self.texts[idx]
        candidate_list = self.candidates[idx]  # 当前样本的10个候选词
        input_ids, attention_masks = [], []
        for candidate in candidate_list:
            encoded = tokenizer_cls(
                text, 
                candidate,
                padding='max_length',
                truncation=True,
                max_length=100,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
        return {
            'input_ids': torch.stack(input_ids),  # 形状: (10, max_len)
            'attention_mask': torch.stack(attention_masks),
            'candidates': candidate_list
        }

# 初始化分类模型和分词器
CLASSIFIER_MODEL_PATH = r"/checkpoint_cls/fold_5"
tokenizer_cls = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
model_cls = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_PATH, num_labels=2).cuda()
model_cls.eval()

# 创建数据集
classification_dataset = CustomDataset(
    test_df['原始诊断名称'].tolist(),
    loaded_candidates,  # 使用 JSON 文件中的候选词
    tokenizer_cls,
    100
)
####################
classification_dataloader = DataLoader(
    classification_dataset,
    batch_size=4,  # 根据GPU内存调整
    shuffle=False,
    collate_fn=lambda batch: {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),  # (batch_size, 10, max_len)
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'candidates': [item['candidates'] for item in batch]
    }
)

# 分类预测
final_predictions = []
with torch.no_grad():
    for batch in tqdm.tqdm(classification_dataloader, desc="分类预测"):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        batch_candidates = batch['candidates']
        
        # 展平维度
        batch_size, num_candidates, max_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_candidates, max_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, max_len)
        
        # 模型预测
        outputs = model_cls(flat_input_ids, flat_attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        probs = probs.view(batch_size, num_candidates)
        
        # 选择最佳候选词
        best_indices = torch.argmax(probs, dim=1).cpu().numpy()
        for i in range(batch_size):
            final_predictions.append(batch_candidates[i][best_indices[i]])

test_df['predicted_term'] = final_predictions
test_df['predicted_id'] = test_df['predicted_term'].map(standard_terms)
########################################
###评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np

# 计算 BLEU 分数
def compute_bleu(predicted_term, reference_term):
    predicted_tokens = list(jieba.cut(predicted_term))
    reference_tokens = list(jieba.cut(reference_term))
    bleu_score = sentence_bleu(
        [reference_tokens],
        predicted_tokens,
        smoothing_function=SmoothingFunction().method1
    )
    return bleu_score

# 计算 ROUGE 分数
def compute_rouge(predicted_term, reference_term):
    predicted_tokens = list(jieba.cut(predicted_term))
    reference_tokens = list(jieba.cut(reference_term))
    predicted_text = " ".join(predicted_tokens)
    reference_text = " ".join(reference_tokens)
    rouge = Rouge()
    scores = rouge.get_scores(predicted_text, reference_text, avg=True)
    return scores

# 计算指标
accuracy = accuracy_score(test_df['映射结果中文同义词'], test_df['predicted_term'])
precision = precision_score(test_df['映射结果中文同义词'], test_df['predicted_term'], average='macro')
recall = recall_score(test_df['映射结果中文同义词'], test_df['predicted_term'], average='macro')
f1 = f1_score(test_df['映射结果中文同义词'], test_df['predicted_term'], average='macro')

# 计算 BLEU 和 ROUGE 分数
bleu_scores = []
rouge_scores = []

for predicted, reference in tqdm.tqdm(zip(test_df['predicted_term'], test_df['映射结果中文同义词']), desc="计算评估指标"):
    bleu = compute_bleu(predicted, reference)
    rouge = compute_rouge(predicted, reference)
    bleu_scores.append(bleu)
    rouge_scores.append(rouge)

# 打印结果
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1 分数: {f1:.4f}")
print(f"平均 BLEU 分数: {np.mean(bleu_scores):.4f}")
print(f"平均 ROUGE-1 F1 分数: {np.mean([score['rouge-1']['f'] for score in rouge_scores]):.4f}")
print(f"平均 ROUGE-2 F1 分数: {np.mean([score['rouge-2']['f'] for score in rouge_scores]):.4f}")
print(f"平均 ROUGE-L F1 分数: {np.mean([score['rouge-l']['f'] for score in rouge_scores]):.4f}")

# 保存结果
SUBMISSION_PATH = r"./submission.csv"
test_df[['原始诊断名称', 'predicted_term', '映射结果中文同义词', 'predicted_id']].to_csv(SUBMISSION_PATH, index=False)
print(f"预测结果已保存至 {SUBMISSION_PATH}")