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

import itertools 
 
# 缓存约束的逆向
def reverse_constrin():
    concurrent_num=40
    save_duration=50
    similarity_threshold=90
    profile_path='../..//data/profile_corpus/system_profile_frgpts_zh.jsonl'
    js=readjsonl(profile_path)
    
    detail_path='../../data/profile_corpus/system_profile_frgpts_zh_rev.jsonl'
    # 断点续跑
    detail_js=readjsonl(detail_path) if os.path.exists(detail_path) else []  # 中间结果
    done_id=[x['system_prompt'] for x in detail_js]
    js=[x for x in js if x['system_prompt'] not in done_id] 
    print(f'done {len(done_id)}, still have {len(js)} data to process...')

    # 锁
    save_lock = threading.Lock()
    append_lock = threading.Lock()
    # 并行函数
    def reverse_func(item):
        new_item=copy.deepcopy(item) # 先copy
        # 比较早的例子没有排序
        new_item['structure_system_prompt']['约束']=sorted(new_item['structure_system_prompt']['约束'])   
        original_constrain=new_item['structure_system_prompt']['约束'] # 获取原始constrain

        prompt=REVERSE_CONSTRAIN_PROMPT%(original_constrain) # 反转的prompt 
        max_retry_times=3
        retry_times=0
        while retry_times<max_retry_times:
            try:
                reverse_cons=request_chatgpt(prompt=prompt) # 获取反转结果        
                # 提取反转结果
                pred_reverse_cons_map=eval(repair_json(reverse_cons))
                pred_reverse_cons_map=clean_surrogates(pred_reverse_cons_map) 
                pred_reverse_cons_map={x['original_cons']:x['reverse_cons'] for x in pred_reverse_cons_map} # 转成字典后面方便处理

                if len(pred_reverse_cons_map)!=len(original_constrain):
                    raise ValueError('pred reverse constrain num is different from original constrain')
        
                # 原始约束的每个约束，根据生成的结果，重新构造映射，因为可能有微小的不同导致映射不上去
                reverse_cons_map=[] # 最终结果存在这里
                all_pred_original_cons=list(pred_reverse_cons_map.keys()) # 预测的所有原始约束
                for original_cons in original_constrain:       
                    # 找到预测结果里和当前要处理的原始约束最相似的约束
                    similarity_result=find_most_similar(original_cons,all_pred_original_cons)       
                    best_pred,best_similarity=similarity_result['candidate'],similarity_result['similarity'] # 最相似的候选和相似度
                    # 如果找不到类似的就报错
                    if best_similarity<=similarity_threshold:
                        raise ValueError(f'similarity too low, retry to generate reverse constrains {cons}, {best_pred}, {best_similarity}')
                    else:
                        # 这样就不会和原始约束对不上了
                        reverse_cons_map.append({'original_cons':original_cons,'reverse_cons':pred_reverse_cons_map[best_pred]})

                # 能到这里应该就没错了
                new_item['reverse_cons_map']=reverse_cons_map
                return new_item
            except Exception as e:
                print(f'\nan exception raised : {e}\n{traceback.format_exc()}')
                sys.exit(1) # 有错误立刻结束程序
             

    # 多线程
    with ThreadPoolExecutor(max_workers=concurrent_num) as executor:
        item_future_list = [executor.submit(reverse_func, item) for item in tqdm(js)]
        
        for item_future in tqdm(concurrent.futures.as_completed(item_future_list), total=len(item_future_list)):
            try:
                new_item=item_future.result()        # 增加数据
                with append_lock:
                    detail_js.append(new_item) 
                # 安全保存
                if len(detail_js)%save_duration==0:              
                    concurrent_savejsonl(detail_path,detail_js,save_lock) 
            except Exception as e: 
                print(f'\nan exception raised in multi-process : {e}\n{traceback.format_exc()}')
                sys.exit(1) # 有错误立刻结束程序

 

if __name__=='__main__':
 
    reverse_constrin()