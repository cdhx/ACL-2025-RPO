# 细粒度评测相关
import json
import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
from src.utils.utils import *
from src.utils.prompt import *
from src.utils.local_service import *
from tqdm import tqdm
import pandas as pd
from datetime import datetime 
 
import concurrent.futures
from tqdm import tqdm

import traceback
import  re
import threading
from json_repair import repair_json
from concurrent.futures import ThreadPoolExecutor, as_completed 
 

# 获取一个query的得分点
def get_evaluate_points(system,query,constrain_list,judge_model_name,max_extract_times=3):
    evaluate_point_list_prompt = SCORE_PROMPT%(system,query)
    extract_times=0
    true=True
    false=False
    similarity_threshold=0.90
    while extract_times<max_extract_times:
        try:
            # 根据情况调用模型
            if type(judge_model_name)==str:
                evaluate_point_detail_list = request_chatgpt(prompt=evaluate_point_list_prompt,model=judge_model_name)
            else:
                evaluate_point_detail_list = request_local(prompt=evaluate_point_list_prompt,ls=ls)
            # 获得触发的约束
            evaluate_point_detail_list = eval(repair_json(evaluate_point_detail_list))
            evaluate_point_detail_list=[x for x in evaluate_point_detail_list if x['评判结果']==True]            
            
            # 检查是否能在原约束里找到对应的
            for eval_idx,eval_point in enumerate(evaluate_point_detail_list):
                # 找到最相似的原约束
                similarity_result=find_most_similar(eval_point['constrain'].strip(),constrain_list)
                best_cand,best_simi=similarity_result['candidate'],similarity_result['similarity']
                if best_simi>=similarity_threshold:
                    evaluate_point_detail_list[eval_idx]['constrain']=best_cand
                else:
                    raise ValueError(f'align with original constrain failed: {similarity_result}') 
            # 如果对齐到原始约束之后，有重复的，也是有问题的
            if len(set([x['constrain'] for x in evaluate_point_detail_list]))!=len(evaluate_point_detail_list):
                raise ValueError('duplicate evaluate points!')

            if len(evaluate_point_detail_list)>0:
                # evaluate_point_list是constrain的列表，evaluate_point_detail_list是评价的细节
                # 按照字典序排序，控制顺序一致
                evaluate_point_detail_list.sort(key=lambda x:x['constrain'])
                # 提取出约束本身，和detail一起返回
                evaluate_point_list=[x['constrain'] for x in evaluate_point_detail_list]
                # 看看有多少要重试的，不用重试的就不打印了
                if extract_times!=0:
                    print(f'success get evaluate points, try {extract_times+1} times')                
                return evaluate_point_list,evaluate_point_detail_list
            else:
                raise ValueError('extract zero evaluate points!')                
        except Exception as e:                         
            extract_times+=1
            print(f'\nextract evaluate point raise an exception: {e}\n{traceback.format_exc()}')
    print(f'retry {max_extract_times} still not work')


# 给定一个回复和所有评分点，给出每个评分点的结果
def judge_response_with_points(judge_config):
    query=judge_config['query']
    response=judge_config['response']
    evaluate_point_list=judge_config['evaluate_point_list']
    judge_model_name=judge_config['judge_model_name']
    source=judge_config['source']

    max_retry = 5
    retry = 0
    true = True
    false = False
    similarity_threshold=0.90
    points_num=len(evaluate_point_list)
    # 如果评测点数量是0
    if points_num==0:
        raise ValueError('there is no evaluate point to evaluate!')
 
    fine_grained_evaluate_prompt = BATCH_POINT_FINE_GRAINED_EVALUATE_PROMPT.format(user_query=query, response=response, constrain_list='\n'.join([str(i+1)+'. '+x for i,x in enumerate(evaluate_point_list)]))
     
    while retry < max_retry:
        try:
            if type(judge_model_name)==str:
                temp_evaluate_result = request_chatgpt(prompt=fine_grained_evaluate_prompt, model=judge_model_name)
            else:
                temp_evaluate_result = request_local(prompt=fine_grained_evaluate_prompt, ls=judge_model_name)
            temp_evaluate_result = temp_evaluate_result.strip()
             
            if temp_evaluate_result.startswith('```json') and temp_evaluate_result.endswith('```'):
                temp_evaluate_result = temp_evaluate_result[7:-3].strip()
            # 可能会触发utf-8的问题             
            temp_evaluate_result = eval(repair_json(temp_evaluate_result))
            # 三个key要有
            if len([point_detail for point_detail in temp_evaluate_result if 'constrain' in point_detail.keys() and '评判理由' in point_detail.keys() and '评判结果' in point_detail.keys()])==len(temp_evaluate_result): 
                # 格式检查，必须有这个key并且必须是TF取值
                if len([x for x in temp_evaluate_result if x['评判结果'] not in [True, False]]) == 0:

                    # 评测内容要对应上 处理每个点,对齐到原始约束上
                    for eval_idx,eval_point in enumerate(temp_evaluate_result):
                        # 找到最相似的原约束
                        similarity_result=find_most_similar(eval_point['constrain'].strip(),evaluate_point_list)
                        best_cand,best_simi=similarity_result['candidate'],similarity_result['similarity']
                        if best_simi>=similarity_threshold:
                            temp_evaluate_result[eval_idx]['constrain']=best_cand
                        # 如果有找不到满足要求的constrain，是有问题的
                        else:
                            raise ValueError(f'align with original constrain failed: {similarity_result}') 
                    # 如果有重复的，也是有问题的
                    if len(set([x['constrain'] for x in temp_evaluate_result]))!=len(temp_evaluate_result):
                        raise ValueError('duplicate evaluate points!')
                    # 用字典序控制约束的顺序一致
                    temp_evaluate_result.sort(key=lambda x:x['constrain'])

                    # 填写数据
                    score = len([x for x in temp_evaluate_result if x['评判结果'] == True])
                    # 看看有多少要重试的，不用重试的就不打印了
                    if retry!=0:
                        print(f'success get evaluate detail, try {retry+1} times')
                    if type(judge_model_name)==str:
                        return {'score': score, 'rate': score/points_num,'source': source, 'judge_model':judge_model_name,'response': response, 'detail': temp_evaluate_result}
                    else:
                        return {'score': score, 'rate': score/points_num, 'source': source, 'judge_model':judge_model_name.served_name,'response': response, 'detail': temp_evaluate_result}
                                
        except Exception as e:
            retry += 1
            print('evaluate parsing error, retrying...',e , temp_evaluate_result)
    return None  # 如果未成功，返回 None
 


if __name__ == "__main__":
    pass 