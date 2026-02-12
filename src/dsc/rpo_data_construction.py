# 解决多约束问题
import json
import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
from src.utils.utils import *
from src.utils.prompt import *
from src.utils.local_service import *
from src.utils.evaluate_utils import *
from src.utils.self_critic import *
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import emoji
import concurrent.futures
from tqdm import tqdm
from src.utils.data_process import *
import re
import traceback
from fuzzywuzzy import fuzz

import threading
import copy
import difflib
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer
import itertools

 
def remove_duplicates(response_eval_list):
    for response in response_eval_list:
        # 用于存储已经遇到的"判断结果"组合
        seen_results = set()
        # 去重后的
        unique_details = []

        for item in response_eval_list:
            # 提取“判断结果”作为元组
            judgement_result = tuple([x['评判结果'] for x in item['detail']])

            # 如果这个结果组合没有见过，保留
            if judgement_result not in seen_results:
                unique_details.append(item)
                seen_results.add(judgement_result)

    return unique_details
# 构造rpo的数据
def rpo_data_construction():
    # 分析现有response中，哪些可以通过逆向约束构造更多偏好对 
    # ======读detail数据========
    eval_file_path_list=[
        "../../data/rpo_data/detail_for_dpo_frgpts_gpt-4o-mini_gpt-4o-mini.json"
    ]
    js=[]
    for eval_file_path in eval_file_path_list:
        js+=readjson(eval_file_path)

    rpo_js=[]
    # =========找出来有区分度的response=============
    for session in tqdm(js,desc='rpo data construction...'):
        # 获取约束反向映射
        reverse_cons_map={cons_map['original_cons']:cons_map['reverse_cons'] for cons_map in session['reverse_cons_map']}
        # 获取结构化的原始system prompt
        original_structured_system_prompt=session['structure_system_prompt']
        # 构造messages history，先不放system，因为是动态的，最后拼接到前面就行了
        messages=[]
        for tid,turn in enumerate(session['infer_history']):
            # 这一轮的query
            messages.append({'from':'human','value':turn['query']})

            # 每个turn所有回复的评分细节
            response_eval_list=turn['finegrain_score']
            # 分数完全一致的留一个就行
            response_eval_list=remove_duplicates(response_eval_list)

            # ============两两组合所有候选，按照True False列表差距排序==============
            reverse_dpo_pair_list=[] # 用于构造rpo数据的基本信息
            for detail1, detail2 in itertools.combinations(response_eval_list, 2):
                # 提取detail中的评判结果 
                difference_count = 0 # 取值不同的点的数量
                differing_elements = [] # 取值不同的点有哪些

                # 判断每个得分点得分是否一致，得到分数差异和差异点
                for (item1, item2) in zip(detail1['detail'], detail2['detail']):
                    result1 = item1['评判结果']
                    result2 = item2['评判结果']

                    # 计算差距
                    if result1 != result2:
                        difference_count += 1
                        differing_elements.append((item1, item2))  # 记录不同的元素
                if len(differing_elements)!=0:
                    # 分别表示第一个，第二个元素，完美情况下的约束是什么样的
                    # TODO, 获取reverse约束写了个bug，映射的key和value都是预测的，original应该是原始约束才对 
                    constrain1=[x['constrain'] if x['评判结果']==True else reverse_cons_map[find_most_similar(x['constrain'],list(reverse_cons_map.keys()))['candidate']] for x in detail1['detail'] ]
                    constrain2=[x['constrain'] if x['评判结果']==True else reverse_cons_map[find_most_similar(x['constrain'],list(reverse_cons_map.keys()))['candidate']] for x in detail2['detail'] ]

                    reverse_dpo_pair_list.append({'difference_count':difference_count,'detail1':detail1,'detail2':detail2,'differing_elements':differing_elements,
                                                  'response1':detail1['response'],'response2':detail2['response'],'constrain1':constrain1,'constrain2':constrain2
                                                  })
            reverse_dpo_pair_list.sort(key=lambda x:x['difference_count'],reverse=True)
            # 构造rpo数据，加入conversation历史，更新system prompt

            # =============构造偏好对================== 
            # 以1为准
            # 以2为准
            # 以原始为准
            # 以原始相反的为准
            # 逐个反转取值有差异的约束
            for reverse_dpo_pair in reverse_dpo_pair_list:
                # =================以1为正例，2为负例================= 
                structured_system_prompt = copy.deepcopy(original_structured_system_prompt)
                structured_system_prompt['约束']=reverse_dpo_pair['constrain1'] # 约束改成以1为完美的
                system_prompt=get_flat_profile(structured_system_prompt)
                rpo_js.append({
                    'conversations': [{'from':'system','value':system_prompt}]+messages,
                    'chosen':{'from':'gpt','value':reverse_dpo_pair['response1']},
                    'rejected':{'from':'gpt','value':reverse_dpo_pair['response2']},
                    'margin':reverse_dpo_pair['difference_count'], 
                    'sid': session['gptId'],
                    'tid': tid+1,
                })
                # =================以2为正例，1为负例================= 
                structured_system_prompt = copy.deepcopy(original_structured_system_prompt)
                structured_system_prompt['约束']=reverse_dpo_pair['constrain2'] # 约束改成以response2为完美的
                system_prompt=get_flat_profile(structured_system_prompt)
                rpo_js.append({
                    'conversations': [{'from':'system','value':system_prompt}]+messages,
                    'chosen':{'from':'gpt','value':reverse_dpo_pair['response2']},
                    'rejected':{'from':'gpt','value':reverse_dpo_pair['response1']},
                    'margin':reverse_dpo_pair['difference_count'], 
                    'sid': session['gptId'],
                    'tid': tid+1,
                })

            # 这一轮结束，messages加上sft response
            messages.append({'from':'gpt','value':turn['sft_response']}) # 更新messages  

 
 
    # 把数据放到llamafactory对应文件夹里，修改dataset_info.json
  
    # *************rpo************
    rpo_js.sort(key=lambda x:x['margin'],reverse=True) 
    rpo_js=[x for x in rpo_js if x['type']!='original']
    # **********dpo基线变种，约束不变，只要有分差就留下**********
    dpo_js=[x for x in rpo_js if x['type']=='original']
     
    # ==============保存写入llamafactory=====  
    # ********rpo，margin>2的********
    rpo_js_gt2=[x for x in rpo_js if x['margin']>2] 
    save_and_update_dataset_info('rpo_js_gt2_margin',rpo_js_gt2)
    save_and_update_dataset_info('rpo_js_gt2',remove_key(rpo_js_gt2,['margin'])) 
    # ********dpo，margin>2的********
    dpo_js_gt2=[x for x in dpo_js if x['margin']>2] 
    save_and_update_dataset_info('dpo_js_gt2_margin',dpo_js_gt2) 
    save_and_update_dataset_info('dpo_js_gt2',remove_key(dpo_js_gt2,['margin'])) 
  
def remove_key(js,key_list):
    new_js=[]
    for idx,item in enumerate(js):
        temp=copy.deepcopy(item)
        for key in key_list:
            del temp[key]
        new_js.append(temp)
    return new_js


# 保存，更新dataset_info
def save_and_update_dataset_info(dataset_name,js):
    # parent_path是data路径下要保存文件的相对路径
    if dataset_name.endswith('.json'):
        dataset_name=dataset_name[:-len('.json')]
    savejson(f'/mnt/workspace/hx/LLaMA-Factory/data/{dataset_name}.json',js)

    # 加入dataset_info
    # 计算max token和长度
    max_token_info=get_max_token(js)

    print('***********************')
    print('data size:',len(js))
    print('qwen 95 percent of data less than ',max_token_info['qwen_len_percent_95'],' tokens. choose ',max_token_info['qwen_max_tokens'],' as cutoff len.')
    print('llama 95 percent of data less than ',max_token_info['llama_len_percent_95'],' tokens. choose ',max_token_info['llama_max_tokens'],' as cutoff len.')
    print('***********************\n\n')

    print(f'saving {len(js)} dpo data at {dataset_name}')

    dataset_info=readjson('/mnt/workspace/hx/LLaMA-Factory/data/dataset_info.json')
    true=True 

    new_info= {
        "file_name": os.path.join(parent_path,f'{dataset_name}.json'),
        "ranking": true,
        "formatting": "sharegpt",
        "num_samples":len(js),
        'qwen_max_tokens':max_token_info['qwen_max_tokens'],
        'llama_max_tokens':max_token_info['llama_max_tokens'],
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
    

    dataset_info[dataset_name] = new_info
    savejson('/mnt/workspace/hx/LLaMA-Factory/data/dataset_info',dataset_info)



if __name__=='__main__':
    config={
        # 本地模型
        'local_model_path': '/mnt/workspace/hx/Models/Qwen2.5-7B-Instruct',
        'data_source':'frgpts',
        'user_profile_path':'../../data/profile_corpus/user_profile.json',
        'total_turn_num':5,
        'concurrent_num':30,
        'save_duration':5,

        'max_critic_times':3,
        'local_sample_num':5,
        'query_model_name':'gpt-4o',
        'response_model_name':'gpt-4o',
        'judge_model_name': 'gpt-4o',
        'critic_model_name': 'gpt-4o',
        'good_enough_score':0.7, # 采样最好的回复低于这个分数要做critic，也是critic要达到目标，不用太高

    }
    config['system_profile_js_path'] = f"../../data/profile_corpus/system_profile_{config['data_source']}.jsonl"
 
    rpo_data_construction() 