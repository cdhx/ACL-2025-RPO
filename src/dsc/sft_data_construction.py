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

import copy
import difflib 
import re

from concurrent.futures import ThreadPoolExecutor, as_completed 

import itertools 

 
# 
def generate_query_sft_data(config): 
    # data_source是system prompt的灵感文本的来源，只做记录用
    # 路径里的data_source可以不管，直接把路径替换成目标数据路径就行 
    data_source=config['data_source']
    # user端的数据
    user_profile_path=config['user_profile_path']
    # system端的数据 需要整理成样例json的格式
    system_profile_js_path = config['system_profile_js_path']    
    system_profile_js=readjsonl(system_profile_js_path)  

    total_turn_num=config['total_turn_num']
    concurrent_num=config['concurrent_num']
    save_duration=config['save_duration']
    sft_sample_num=config['sft_sample_num']
    max_critic_times=config['max_critic_times']
    
    query_model_name=config['query_model_name']
    response_model_name=config['response_model_name']
    judge_model_name=config['judge_model_name']
    critic_model_name=config['critic_model_name']
    good_enough_score=config['good_enough_score']

    # detail_path是生成结果存储的路径
    detail_path=f'../../data/rpo_data/detail_for_sft_{data_source}_{query_model_name}_{response_model_name}.json'  

    # 断点续跑，如果已经存在detail_path对应的文件，就load进来去掉system_profile_js里已经处理过的
    if os.path.exists(detail_path):
        detail_js=readjson(detail_path)  
        detail_js=[x for x in detail_js if len(x['sft_history'])!=0] # 有一些请求失败的空[]
        done_id=[x['inspired_corpus'] for x in detail_js]
        system_profile_js=[x for x in system_profile_js if x['inspired_corpus'] not in done_id][:1000]
    else:
        detail_js = []  # 中间结果

    print('Load with : ',len(detail_js), ' data, still have ',len(system_profile_js),' to process...')
    
    # 每个system配对一个user的profile
    user_js=readjson(user_profile_path) 
    random.shuffle(user_js)
    for index,item in enumerate(system_profile_js): 
        system_profile_js[index]['user_profile']=user_js[index]['profile'] 
        system_profile_js[index]['user_profile_inspired_corpus']=user_js[index]['persona']
    # 锁
    import threading
    lock = threading.Lock()
     
    def generate_sft_data_func(item): 
        nonlocal detail_js
        detail_item=copy.deepcopy(item)

        system=detail_item['system_prompt']         
        user_profile=detail_item['user_profile']
        # user_profile=detail_item['user_profile_inspired_corpus']

        message=[{'role':'system','content':system}] 

        detail_item['sft_history']=[]
        detail_item['query_list']=[]
        detail_item['system_source']=data_source
        
        characteristic=random.sample(CHARACTERISTIC_LIST,10) # 随机选10个人格        
        # 一轮至少4次请求,query(可能加一次小改写),得分点,response有n次,评估有n次,2n次self-critic(response,评估)
        for turn_idx in tqdm(range(total_turn_num)):
            # =============生成query=================
            # 尝试生成没有重复的query
            past_query_list=[x for x in detail_item['query_list']]
            similar_message = None # 重置这个变量
            # 获取query
            get_query_prompt=USER_QUERY_POMPT.format(characteristic=characteristic,user_profile=user_profile,system_profile=system,history=message[max(1,turn_idx-10):]) # system prompt不算历史,作为历史的只留最后五轮（10个message元素）                        
            query=request_chatgpt(model=query_model_name,prompt=get_query_prompt,temperature=1.0)     
            query=remove_emoji(query).strip('"').strip("'") # 用户query有emoji是不正常的           
            # 控制query不能和之前重复，返回值是重复的query是什么，或者None
            similar_message = detect_similar_query(query,past_query_list)

            # 如果有相似的query,就对query做重写,很明确很简单的任务,基本不需要retry
            if similar_message:
                rephrase_query_prompt=REPHRASE_USER_QUERY_PROMPT.format(similar_message=[similar_message,query[:8]],query=query)
                query=request_chatgpt(model=query_model_name,prompt=rephrase_query_prompt,temperature=1.0)
                print(f'rephrase query, similar message: {detect_similar_query(query,past_query_list)}')
            # 打印query和characteristic
            print(f'turn {turn_idx+1},  characteristic:{characteristic[:3]}...,query: {query}')
            
            # ==============采样多个response=================
            # 拼接system prompt得分点到query里面
            evaluate_point_list,evaluate_point_detail_list = get_evaluate_points(system,query,item['structure_system_prompt']['约束'],judge_model_name)
            message.append({'role':'user','content':query+'\n请你务必在回答中记得遵循这些约束:\n'+get_plain_eval_point(evaluate_point_list)}) 
            
            # 多次采样，直接改n
            sft_response_list=request_chatgpt(prompt=message[-10:], temperature=1.0,n=sft_sample_num)
            # print('sft_response_list:', sft_response_list)
 
            # batch评测 
            judge_config={'query':query,'evaluate_point_list':evaluate_point_list,'evaluate_point_detail_list':evaluate_point_detail_list,'judge_model_name':judge_model_name,'source':response_model_name}            
            eval_detail_list=batch_evaluate(judge_config,sft_response_list) # 获取评分结果
            eval_detail_list.sort(key=lambda x:x['rate'],reverse=True) # 排序
        
            # ================对最好的例子做critic===================
            # 当前最好的response对应的评测detail
            current_best_detail=eval_detail_list[0]
            # 现在只做sft: good_enough是一个比较低的值，这个都达不到有点太低了
            if current_best_detail['rate']<good_enough_score:
                print(f"turn {turn_idx+1}, un-perfect response {current_best_detail['score']}  / {len(current_best_detail['detail'])}")
                # 持续critic，基于目前最好的（包括critic后的）一直更新
                critic_detail_list=continuously_self_critic(current_best_detail['response'],query,current_best_detail,critic_model_name,judge_model_name,threshold=good_enough_score,max_critic_times=max_critic_times)
                # critic后的response插到最前面，没变好也无所谓先留下来
                eval_detail_list=critic_detail_list+eval_detail_list # critc的结果合并进来
                eval_detail_list.sort(key=lambda x:x['rate'],reverse=True) # 排序
                current_best_detail=eval_detail_list[0] # 更新最好的回复

            
            # 如果critic之后还是不行，结束session
            if current_best_detail['rate']<good_enough_score:
                print(f'End session with {turn_idx} turn, retry {max_critic_times} times stil generate un-perfect response')
                break
            # 有足够好的response
            else:
                sft_response=current_best_detail['response']     
                print(f'turn {turn_idx+1}, perfect response ! {sft_response[:30]}...')
                      
            # 保留的query不能每次都拼接system prompt
            message=message[:-1]
            message.append({'role':'user','content':query}) 
            message.append({'role':'assistant','content':sft_response})          
            
            detail_item['sft_history'].append({'query':query,'sft_response':sft_response,'evaluate_point_list':evaluate_point_list})
            detail_item['query_list'].append(query)
        print(f"finised {item['gptId']}")        
        now = datetime.now() # 获取当前时间 
        readable_timestamp = now.strftime("%Y-%m-%d %H:%M:%S") # 格式化为可读的字符串 
        detail_item['timestamp']=readable_timestamp 

        # 添加新数据
        with lock:
            try:            
                detail_js.append(detail_item) 
                if len(detail_js)%save_duration==0: 
                    detail_js=clean_surrogates(detail_js) # 处理代理对   
                    savejson(detail_path,detail_js)  
                    print('saved with ',len(detail_js),' data!') 
            except Exception as e:
                print(f'exception raise when save result: {e}')

    # session级别并行
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_num) as executor: 
        item_future_list = [executor.submit(generate_sft_data_func, item) for item in system_profile_js]
        
        for item_future in tqdm(concurrent.futures.as_completed(item_future_list), total=len(item_future_list)):
            try:
                result = item_future.result(timeout=180)
            except Exception as e:
                print(f'\nan exception raised : {e}\n{traceback.format_exc()}')
                sys.exit(1) # 有错误立刻结束程序
             

    detail_js=clean_surrogates(detail_js) # 处理代理对   
    savejson(detail_path,detail_js) 
    return detail_path



if __name__=='__main__':
    config={
        # 本地模型 
        'data_source':'frgpts',
        'user_profile_path':'/mnt/workspace/hx/ACL2025/data/profile_corpus/user_profile.json',
        'total_turn_num':5,
        'concurrent_num':20,
        'save_duration':10,
        
        'max_critic_times':3,
        'sft_sample_num':5, # 现在改成通过n控制了，这部分token增加不大，但是每个都要evaluate，这部分会增加一点开销
        'query_model_name':'gpt-4o-mini', 
        'response_model_name':'gpt-4o-mini', 
        'judge_model_name': 'gpt-4o-mini',
        'critic_model_name': 'gpt-4o-mini',
        'good_enough_score':0.7, # 采样最好的回复低于这个分数要做critic，也是critic要达到目标，不用太高

    }
    config['system_profile_js_path'] = f"../../data/profile_corpus/system_profile_{config['data_source']}_zh_rev.jsonl"   
    generate_query_sft_data(config)