import requests
import json
import os
import re
import tiktoken 
import random
import itertools 
import logging
import copy
from tqdm import tqdm 
from openai import OpenAI
import uuid
import sys

# from transformers import AutoTokenizer
  
import os
import dashscope
import random

from fuzzywuzzy import fuzz

from json_repair import repair_json
import copy
import difflib 
import re
import emoji

import http.client
import json 

# 处理代理对
def clean_surrogates(data):
    if isinstance(data, str):
        # 将代理对转换为实际的Unicode字符
        return data.encode('utf-16', 'surrogatepass').decode('utf-16')
    elif isinstance(data, list):
        return [clean_surrogates(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_surrogates(v) for k, v in data.items()}
    return data
def generate_short_uuid(length=16):
    # 生成一个UUID并转为字符串
    full_uuid = str(uuid.uuid4())
    # 截取前length个字符
    short_uuid = full_uuid.replace("-", "")[:length]
    return short_uuid
 
 

def get_plain_eval_point(evaluate_point_list):
    return '\n'.join([str(i+1)+'. '+x for i,x in enumerate(evaluate_point_list)])
 
# 一个session的对话历史里是否有类似的query出现
def has_similar_query(your_query_list): 
            
    max_token=8
    # query第一个标点之前，10个token以内
    past_query_list=[re.split(r"""[,，.。;；:：“”‘’"'!！?？]+""", x)[0][:max_token] for x in your_query_list]
        
    # 生成所有两两组合
    combinations = list(itertools.combinations(past_query_list, 2))

    for q1,q2 in combinations:
        similarity = fuzz.ratio(q1, q2)
        # 至少一个相似度过高的
        if similarity>60:                
            return True
    return False

def detect_similar_query(query,past_query_list,max_token=8):
    # 控制query不能重复,第一个标点之前，8个token以内
    query = query.strip('"').strip("'")
    past_query_list = [x.strip('"').strip("'") for x in past_query_list]
    past_query_list=[re.split(r"""[,，.。;；:：“”‘’"'!！?？]+""", x)[0][:max_token] for x in past_query_list]
    query=re.split(r"""[,，.。;；:：“”‘’"'!！?？]+""", query)[0][:max_token]
     
    for past_q in past_query_list:
        similarity = fuzz.ratio(query, past_q)
        # 前三个一模一样也不行
        if similarity>60 or query[:3]==past_q[:3]:  
            print(f'similar query, {query} ==== {past_q} ')
            return past_q
    return None
 

# 计算一个list里和目标string的相似度，返回最相似的哪个
def find_most_similar(target,candidate_list):
    # result=[{"candidate":cand,"target":target,"similarity":fuzz.ratio(target, cand)} for cand in candidate_list]
    # result=sorted(result,key=lambda x:x['similarity'],reverse=True)
    # return result[0]

    best_cand=None
    best_similarity=0
    for cand in candidate_list:
        similarity = fuzz.ratio(target, cand)
        if best_similarity<similarity:
            best_similarity=similarity
            best_cand = cand
    
    return {"candidate":best_cand,"target":target,"similarity":best_similarity} 
  

def request_chatgpt(prompt='hi',model='gpt-4o-mini',temperature=0,max_retry=3,n=1):
     
    url = "YOU_REQUEST_URL"
    # 自己的
    token ="YOU_API_KEY"    

    # messages格式
    if type(prompt)==list and 'role' in prompt[0].keys() and 'content' in prompt[0].keys():
        messages=prompt
    # str格式
    if type(prompt)==str:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    payload = {
    'model': model,
    'messages': messages,
    'temperature':temperature,
    'n':n,
    }

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    retry=0
    while retry<max_retry:
        try: 
            response = requests.request("POST", url, headers=headers, json=payload) 
 
            if response.status_code==200: 
                null=None 
                if n==1:      
                    return eval(response.text)['data']['response']['choices'][0]['message']['content']                    
                else: 
                    return [x['message']['content'] for x in eval(response.text)['data']['response']['choices']]

        except Exception as e:
            retry+=1
            print(f'Retrying {retry+1}...  ', e)

 
 
def readjson(path):
    '''读取json,json_list就返回list,json就返回dict'''
    with open(path, mode='r', encoding='utf-8') as load_f:
        data_ = json.load(load_f)
    return data_
 
def savejson(file_name, json_info, indent=4):
    if file_name.endswith('.json'):
        file_name = file_name[:-5]
    temp_file = f'{file_name}_temp.json'
    final_file = f'{file_name}.json'    
    # 将数据写入临时文件
    with open(temp_file, mode='w', encoding='utf-8') as fp:
        json.dump(json_info, fp, indent=indent, ensure_ascii=False)
    # 使用原子操作重命名文件
    os.replace(temp_file, final_file)  # 在大多数操作系统上，这是原子操作 


def savejsonl(file_name, data):
    if file_name.endswith('.jsonl'):
        file_name = file_name[:-6]
    temp_file = f'{file_name}_temp.jsonl'
    final_file = f'{file_name}.jsonl' 
    # 将数据写入临时文件
    with open(temp_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # 使用原子操作重命名临时文件为最终文件
    os.replace(temp_file, final_file)


def concurrent_savejson(detail_path,detail_js,save_lock):              
    try:        
        with save_lock:             
            detail_js=clean_surrogates(detail_js) # 处理代理对   
            savejson(detail_path,detail_js)  
            print('saved with ',len(detail_js),' data!') 
    except Exception as e:                                       
        print(f'exception raise when save result: {e}')
        # 保存出问题可以停了，因为有问题的数据已经append进来了,后面的算出来也一个都保存不下来
        raise ValueError("bad data in json!")     

def concurrent_savejsonl(detail_path,detail_js,save_lock):              
    try:        
        with save_lock:             
            detail_js=clean_surrogates(detail_js) # 处理代理对   
            savejsonl(detail_path,detail_js)  
            print('saved with ',len(detail_js),' data!') 
    except Exception as e:                                       
        print(f'exception raise when save result: {e}')
        # 保存出问题可以停了，因为有问题的数据已经append进来了,后面的算出来也一个都保存不下来
        raise ValueError("bad data in json!")     

def readjsonl(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results
 
def remove_emoji(string):
    return emoji.replace_emoji(string, replace='')


def has_common_substring(main_string, substrings):
    for substring in substrings:
        if substring in main_string:
            return True
    return False
 
# 将结构化的system profile转成字符串
def get_flat_profile(system_prompt):
    skill_text='\n'.join([str(i+1)+'. '+x for i,x in enumerate(system_prompt['技能'])])
    constrain_text='\n'.join([str(i+1)+'. '+x for i,x in enumerate(system_prompt['约束'])])
    flat_profile=f"## profile:\n名称: {system_prompt['名称']}\n描述: {system_prompt['描述']}\n\n## 技能:\n{skill_text}\n\n## 约束:\n{constrain_text}"
    return flat_profile
     
if __name__ == "__main__": 
    pass
 