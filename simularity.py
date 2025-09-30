import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import re

def preprocess_text(text):
    """文本預處理"""
    if pd.isna(text) or text is None:
        return ""
   
    text = str(text).strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
   
    return text.lower()

def extract_category_from_filename(filename):
    """從檔名提取類別"""
    if not filename:
        return "unknown"
    filename = filename.lower()
    if "single" in filename:
        return "single"
    elif "comp" in filename:
        return "competition"
    elif "coop" in filename:
        return "cooperation"
    else:
        return "unknown"

def load_json_sentences(json_files):
    """載入JSON檔案中的句子"""
    sentences = []
   
    for json_file in json_files:
        print(f"Loading {json_file}...")
       
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
           
            json_name = os.path.basename(json_file).replace('_description_json.json', '')
           
            for item in data:
                if 'class' in item and item['class']:
                    processed_text = preprocess_text(item['class'])
                    if processed_text:
                        sentences.append({
                            'id': f"json_{len(sentences)}",
                            'source': 'json',
                            'file': json_name,
                            'category': extract_category_from_filename(item.get('image1', '')),
                            'text': item['class'],
                            'processed': processed_text
                        })
                       
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
   
    print(f"Loaded {len(sentences)} sentences from JSON files")
    return sentences

def load_csv_sentences(csv_file):
    """載入CSV檔案中的生成句子"""
    print(f"Loading {csv_file}...")
   
    sentences = []
   
    try:
        df = pd.read_csv(csv_file)
       
        for _, row in df.iterrows():
            if 'generated' in row and pd.notna(row['generated']):
                processed_text = preprocess_text(row['generated'])
                if processed_text:
                    sentences.append({
                        'id': f"csv_{len(sentences)}",
                        'source': 'csv',
                        'file': 'generated',
                        'category': row.get('category', ''),
                        'style': row.get('style', ''),
                        'strategy': row.get('strategy', ''),
                        'root_dir': row.get('root_dir', ''),
                        'text': row.get('generated', ''),
                        'processed': processed_text
                    })
       
        print(f"Loaded {len(sentences)} generated sentences from CSV file")
        return sentences
       
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return []

def calculate_similarities(json_sentences, csv_sentences):
    """計算JSON和CSV句子之間的相似度"""
   
    # 準備所有句子的處理文本
    all_processed = [s['processed'] for s in json_sentences] + [s['processed'] for s in csv_sentences]
   
    print("Calculating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
   
    tfidf_matrix = vectorizer.fit_transform(all_processed)
   
    print("Calculating cosine similarity...")
   
    # 分割矩陣：JSON部分和CSV部分
    json_matrix = tfidf_matrix[:len(json_sentences)]
    csv_matrix = tfidf_matrix[len(json_sentences):]
   
    # 計算JSON vs CSV的相似度
    similarity_matrix = cosine_similarity(json_matrix, csv_matrix)
   
    return similarity_matrix

def save_similarity_results(similarity_matrix, json_sentences, csv_sentences, timestamp):
    """儲存相似度結果"""
   
    # 創建結果列表
    results = []
   
    for i, json_sent in enumerate(json_sentences):
        for j, csv_sent in enumerate(csv_sentences):
            similarity = similarity_matrix[i][j]
           
            results.append({
                'json_id': json_sent['id'],
                'csv_id': csv_sent['id'],
                'similarity': similarity,
                'json_file': json_sent['file'],
                'json_category': json_sent['category'],
                'csv_category': csv_sent['category'],
                'csv_style': csv_sent.get('style', ''),
                'csv_strategy': csv_sent.get('strategy', ''),
                'csv_root_dir': csv_sent.get('root_dir', ''),
                'json_text': json_sent['text'],
                'csv_text': csv_sent['text']
            })
   
    # 轉換為DataFrame並排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('similarity', ascending=False)
   
    # 儲存結果
    output_file = f"sentence_similarities_{timestamp}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
   
    print(f"相似度結果已儲存至: {output_file}")
   
    # 顯示統計
    print(f"\n=== 統計摘要 ===")
    print(f"JSON句子數: {len(json_sentences)}")
    print(f"CSV句子數: {len(csv_sentences)}")
    print(f"總比較對數: {len(results)}")
    print(f"平均相似度: {results_df['similarity'].mean():.4f}")
    print(f"最高相似度: {results_df['similarity'].max():.4f}")
    print(f"最低相似度: {results_df['similarity'].min():.4f}")
   
    # 相似度分布
    high_sim = len(results_df[results_df['similarity'] > 0.7])
    medium_sim = len(results_df[(results_df['similarity'] >= 0.4) & (results_df['similarity'] <= 0.7)])
    low_sim = len(results_df[results_df['similarity'] < 0.4])
   
    print(f"高相似度 (>0.7): {high_sim} 對")
    print(f"中等相似度 (0.4-0.7): {medium_sim} 對")
    print(f"低相似度 (<0.4): {low_sim} 對")
   
    # 顯示前10個最相似的句子對
    print(f"\n=== 前10個最相似的句子對 ===")
    for idx, row in results_df.head(10).iterrows():
        print(f"{idx+1}. 相似度: {row['similarity']:.4f}")
        print(f"   JSON({row['json_file']}-{row['json_category']}): {row['json_text'][:60]}...")
        print(f"   CSV({row['csv_category']}-{row['csv_style']}): {row['csv_text'][:60]}...")
        print()
   
    return output_file, results_df

def main():
    # 設定檔案路徑
    JSON_FILES = [
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/formal_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/lively_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/mixed_description_json.json"
    ]
   
    # 請修改為你的CSV檔案路徑
    CSV_FILE = "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/natural_generation_results_20250908_023610.csv"  # 請修改為實際檔案名
   
    print("=== 簡化版句子相似度計算 ===")
    print("只計算JSON vs CSV的相似度")
    print("="*40)
   
    # 載入句子
    json_sentences = load_json_sentences(JSON_FILES)
    csv_sentences = load_csv_sentences(CSV_FILE)
   
    if len(json_sentences) == 0 or len(csv_sentences) == 0:
        print("沒有找到句子，請檢查檔案路徑")
        return
   
    print(f"準備計算 {len(json_sentences)} 個JSON句子 vs {len(csv_sentences)} 個CSV句子的相似度")
   
    # 計算相似度
    similarity_matrix = calculate_similarities(json_sentences, csv_sentences)
   
    # 儲存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file, results_df = save_similarity_results(similarity_matrix, json_sentences, csv_sentences, timestamp)
   
    print(f"\n=== 完成 ===")
    print(f"相似度檔案已儲存: {output_file}")
    print("檔案包含每對句子的相似度數值！")

if __name__ == "__main__":
    main()
