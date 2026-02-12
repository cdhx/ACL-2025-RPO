# 2025.06.04 过滤敏感词

# 读取文件夹里所有的json和jsonl文件
# 读取xlsx敏感关键词文件，每行是一个词只有一行，第一行是title不用管
# 对比每个json里的内容，如果包含敏感词，则打印出来
import os
import json 
from tqdm import tqdm


# 读取/mnt/workspace/hx/RPO/src/敏感关键词.xlsx
import pandas as pd
df = pd.read_excel('/mnt/workspace/hx/RPO/src/敏感关键词.xlsx')
sensitive_words = df['关键词'].tolist()
sensitive_words = list(set(sensitive_words))
sensitive_words.append('习近平')
# 依次读取'/mnt/workspace/hx/RPO'这个文件夹的所有文件里，包含子文件夹
file_path_list=[]
for root, dirs, files in os.walk('/mnt/workspace/hx/RPO/data'):
    for file in tqdm(files):
        if file.endswith('.json') or file.endswith('.jsonl'):
            file_path = os.path.join(root, file)            
            # 读取json文件
            if file.endswith('.json'):
                file_path_list.append(file_path)

for file_path in file_path_list:    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)        
        for item in data:
            for word in sensitive_words:
                if word in str(item):
                    print()
                    print(file_path)
                    print(str(item)[:1000])
