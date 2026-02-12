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
  

# 生成query，多个response，作为dpo的数据，不搞sft这里
def generate_query_dpo_data(config): 
    # data_source是system prompt的灵感文本的来源，只做记录用
    # 路径里的data_source可以不管，直接把路径替换成目标数据路径就行
    local_model_path=config['local_model_path']
    data_source=config['data_source'] 
 
    concurrent_num=config['concurrent_num']
    save_duration=config['save_duration']
    local_sample_num=config['local_sample_num']    
    max_critic_times=config['max_critic_times']
    
    query_model_name=config['query_model_name']
    response_model_name=config['response_model_name']
    judge_model_name=config['judge_model_name']
    critic_model_name=config['critic_model_name']
    good_enough_score=config['good_enough_score']

    # 本地模型启动
    print('===============Staring local model=================')
    ls=prepare_local_service(local_model_path)
    # print('==============Local model stated=================\n',request_local('hello',ls))
    

    # sft路径
    sft_detail_path=f'../../data/rpo_data/detail_for_sft_{data_source}_{query_model_name}_{response_model_name}.json'
    # detail_path是生成结果存储的路径
    detail_path=f'../../data/rpo_data/detail_for_dpo_{data_source}_{query_model_name}_{response_model_name}.json'  



    # 断点续跑，如果已经存在detail_path对应的文件，就load进来去掉sft_js里已经处理过的
    sft_js=readjson(sft_detail_path) # 要采样的sft数据
    if os.path.exists(detail_path):
        detail_js=readjson(detail_path)   
        detail_js=[x for x in detail_js if len(x['sft_history'])!=0] # 有一些请求失败的空[]
        done_id=[x['inspired_corpus'] for x in detail_js]
        sft_js=[x for x in sft_js if x['inspired_corpus'] not in done_id]
    else:
        detail_js = []  # 中间结果

    print('Load with : ',len(detail_js), ' data, still have ',len(sft_js),' to process...')
     
    # 锁
    import threading
    append_lock = threading.Lock()
    save_lock = threading.Lock()
    def generate_dpo_data_func(item): 
        nonlocal detail_js # 全局最终结果
        detail_item=copy.deepcopy(item) # 这个session的信息

        system=detail_item['system_prompt']          

        message=[{'role':'system','content':system}] 

        detail_item['infer_history']=[] # 采样结果和评估放在这里

        for turn_idx,turn in tqdm(enumerate(item['sft_history'])): 
            query=turn['query']
            golden_response=turn['sft_response']
            message.append({'role':'user','content':query}) 
            # ==============采样多个response=================  
            # 多次采样 
            local_response_list=request_local(message,ls,temperature=1.0,n=local_sample_num)
            # print('sample done')
            # batch评测 
            evaluate_point_list=turn['evaluate_point_list'] # 得分点
            judge_config={'query':query,'evaluate_point_list':evaluate_point_list,'judge_model_name':judge_model_name,'source':response_model_name}            
            eval_detail_list=batch_evaluate(judge_config,local_response_list) # 获取评分结果
            eval_detail_list.sort(key=lambda x:x['rate'],reverse=True) # 排序
            # print('evaluate done')
        
            # ================对最好的例子做critic===================
            # 当前最好的response对应的评测detail
            current_best_detail=eval_detail_list[0]
            #　目前没有足够response，做critic，这个不是为了得到够好的sft response，而是为了让不同response之间有点差距，如果最高的都不够高，就没有几个分数等级了
            if current_best_detail['rate']<good_enough_score:
                print(f"turn {turn_idx+1}, un-perfect response {current_best_detail['score']}  / {len(current_best_detail['detail'])}")
                # 重新生成，还是critic
                critic_detail_list=continuously_self_critic(current_best_detail['response'],query,current_best_detail,critic_model_name,judge_model_name,threshold=good_enough_score,max_critic_times=max_critic_times)
                # critic后的response插到最前面,如果一点都没变好就不用
                eval_detail_list=critic_detail_list+eval_detail_list # critc的结果合并进来
                eval_detail_list.sort(key=lambda x:x['rate'],reverse=True) # 排序
                current_best_detail=eval_detail_list[0] # 更新最好的回复
                      
            # 采成什么样就是什么样，不用end session
            points_num=len(evaluate_point_list) # 得分点总数
            score_list=[x['score'] for x in eval_detail_list] # 分数列表
            rate_list=[x['score']/points_num for x in eval_detail_list] # 得分率列表
            # 下一轮
            message.append({'role':'assistant','content':golden_response})          
            
            detail_item['infer_history'].append({'query':query,'sft_response':golden_response,'points_num':points_num,'score_list':score_list,'rate_list':rate_list,'evaluate_point_list':evaluate_point_list,'finegrain_score':eval_detail_list})
            detail_item['query_list'].append(query)
                
        now = datetime.now() # 获取当前时间 
        readable_timestamp = now.strftime("%Y-%m-%d %H:%M:%S") # 格式化为可读的字符串 
        detail_item['timestamp']=readable_timestamp 
        detail_item['data_source']=data_source
        detail_item['local_model_path']=local_model_path
        detail_item['judge_model_name']=judge_model_name

        # 添加新数据 
        with append_lock:
            detail_js.append(detail_item) 
        # 保存
        if len(detail_js)%save_duration==0:           
            detail_js=clean_surrogates(detail_js) # 处理代理对       
            concurrent_savejson(detail_path,detail_js,save_lock) 
 

    # session级别并行
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_num) as executor: 
        item_future_list = [executor.submit(generate_dpo_data_func, item) for item in sft_js]
        
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
        'local_model_path': '/mnt/workspace/hx/Models/Qwen2.5-7B-Instruct',
        'data_source':'frgpts',  
        'concurrent_num':20,
        'save_duration':5,
        
        'max_critic_times':3,
        'local_sample_num':5,
        'query_model_name':'gpt-4o-mini', 
        'response_model_name':'gpt-4o-mini', 
        'judge_model_name': 'gpt-4o-mini',
        'critic_model_name': 'gpt-4o-mini',
        'good_enough_score':0.7, # 采样最好的回复低于这个分数要做critic，也是critic要达到目标，不用太高

    } 
    generate_query_dpo_data(config)
    