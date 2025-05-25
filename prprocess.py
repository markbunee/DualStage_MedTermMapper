import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# 读取Excel文件
file_path = r"D:\ASUS\CHIP2021-Task3-Top3-master\final\origin_dataset\data_train.xlsx"  
df = pd.read_excel(file_path)

# 提取第二列和第四列
extracted_df = df.iloc[:, [1, 3]].copy()
extracted_df.columns = ["text_a", "text_b"]

# 预处理函数：清除空格、中英文括号、逗号等符号
def preprocess(text):
    if pd.isna(text):
        return text
    # 清除空格
    text = text.strip()
    # 清除中英文括号及其中的内容
    text = re.sub(r'（[^）]*）', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    # 清除逗号等符号
    text = re.sub(r'[，,]', '', text)
    return text

# 应用预处理
extracted_df["text_a"] = extracted_df["text_a"].apply(preprocess)
extracted_df["text_b"] = extracted_df["text_b"].apply(preprocess)

# 添加标签列（原始诊断名称和映射结果中文同义词对应关系为标签1）
extracted_df["label"] = 1

# 从Excel文件中提取候选词（映射结果中文同义词列）
candidate_words = df.iloc[:, 3].drop_duplicates().tolist()

# 为每行生成两个不匹配的词（基于相似度）
non_matching_pairs = []
for index, row in extracted_df.iterrows():
    text_a = row["text_a"]
    
    # 过滤掉当前行的映射结果中文同义词
    filtered_candidates = [word for word in candidate_words if preprocess(word) != row["text_b"]]
    
    if not filtered_candidates:
        continue  # 如果没有候选词，则跳过
    
    # 随机选择10个候选词（如果候选词不足10个，则选择所有）
    selected_candidates = random.sample(filtered_candidates, min(10, len(filtered_candidates)))
    
    # 计算TF-IDF向量
    tfidf_vectorizer = TfidfVectorizer()
    texts = [text_a] + selected_candidates
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 计算相似度
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    
    # 找到相似度最低的两个词
    sorted_indices = similarities.argsort()
    lowest_indices = sorted_indices[:2]
    
    for idx in lowest_indices:
        non_matching_word = preprocess(selected_candidates[idx])
        non_matching_pairs.append({
            "text_a": text_a,
            "text_b": non_matching_word,
            "label": 0
        })

# 将标签0的数据添加到DataFrame
non_matching_df = pd.DataFrame(non_matching_pairs)
extracted_df = pd.concat([extracted_df, non_matching_df], ignore_index=True)

# 保存为UTF-8编码的CSV文件
output_path = r"D:\ASUS\CHIP2021-Task3-Top3-master\final\dataset\train_pair_dataset.csv"
extracted_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"处理完成！已保存到 {output_path}")