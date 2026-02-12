# 生成profile的放在这里

import json
import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
from src.utils.utils import *
from src.utils.prompt import * 
from tqdm import tqdm
import concurrent
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import traceback
import threading
from datetime import datetime 


# 给定persona的数据生成user的profile
def generate_user_profile_given_persona():
    # 给定persona的数据生成user的profile
    def generate_profile(item):
        inspiration = item['persona']
        prompt = USER_PROFILE_PROMPT.format(inspiration=inspiration)
        profile = request_chatgpt(prompt)
        return {'persona': inspiration, 'profile': profile}
 
    
    js = readjsonl('../../data/profile_corpus/persona.jsonl')

    profile_list = readjson('../../data/profile_corpus/user_profile.json')
    # 
    done_persona=[x['persona'] for x in profile_list]
    js=[x for x in js if x['persona'] not in done_persona] 
    print('Load with : ',len(profile_list), ' data, still have ',len(js),' to process...')

    # 设置线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_profile, item): item for item in js}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)): 
            profile_list.append(future.result())
            
            # 每处理50个保存一次
            if len(profile_list) % 50 == 0:
                savejson('../../data/user_profile', profile_list)
                print('saved with ',len(profile_list),' data!') 

    # 保存剩余的结果
    savejson('../../data/user_profile', profile_list)
 

 
# 用gptstore的数据作为灵感文本生成system的profile   
def generate_system_profile_given_gpts(data_source="gptstore_all",advance_model_name='gpt-4o-mini',language='zh'):
    # 用gptstore的数据作为灵感文本生成system的profile 
    save_path=f'/mnt/workspace/hx/ACL2025/data/profile_corpus/system_profile_{data_source}_{language}.jsonl'
    gpts_source_path=f'/mnt/workspace/hx/ACL2025/data/profile_corpus/{data_source}.json' # 灵感文本数据，system的简单profile即可，例如name+description

    concurrent_num=80 # 并发数
    save_duration=100 # 保存间隔 # 用什么模型 
    
    lock = threading.Lock()
 
    gpts_source_js=readjson(gpts_source_path)
 
    # 断点续跑
    if os.path.exists(save_path):
        js=readjsonl(save_path)
    else:
        js=[]  
    
    done_gpts_inspired=[x['inspired_corpus'] for x in js]
    gpts_source_js=[x for x in gpts_source_js if x['inspired_corpus'] not in done_gpts_inspired]
    print(f'done {len(done_gpts_inspired)}, still have {len(gpts_source_js)} to process...')

    def func_gpts_profile(item): 
        max_retry=3
        retry=0
        inspire_corpus=item['inspired_corpus']
        if language=='zh':            
            prompt=PROFILE_AND_CONSTRAIN_GIVEN_PROFILE_PROMPT.format(system_inspired_corpus=inspire_corpus,initial_ontology=INIT_ONTOLOGY) 
        if language=='en':
            raise ValueError('language en not implemented!')
        result_js=None
        while retry<max_retry:
            try:
                res=request_chatgpt(prompt=prompt,model=advance_model_name,temperature=0.7)          
                res=eval(repair_json(res))                      
                # 约束部分按字典序排列                 
                res['约束']=sorted(res['约束'])                   

                now = datetime.now() # 获取当前时间 
                readable_timestamp = now.strftime("%Y-%m-%d %H:%M:%S") # 格式化为可读的字符串  
                result_js={'system_prompt':get_flat_profile(res),'structure_system_prompt':res,'inspired_corpus':inspire_corpus,'source':data_source,'gen_model':advance_model_name,'timestamp':readable_timestamp } 
                result_js.update(item)
                break
            except Exception as e:
                print(f'\nan exception raised when request or parsing: {e}\n{traceback.format_exc()}')
                retry+=1 
        if result_js is None:
            print(f'retry {max_retry} still error')
        return result_js
        
    # 设置线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_num) as executor:
        futures = {executor.submit(func_gpts_profile, item) for item in gpts_source_js}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):                    
            try: 
                result=future.result(timeout=180)
                with lock:
                    if result is None:
                        continue
                    js.append(result)
                    # 每处理50个保存一次
                    if len(js) % save_duration == 0:                    
                        js=clean_surrogates(js)
                        savejsonl(save_path,js)
                        print('save with ',len(js),' data')
            except TimeoutError as e:   
                print(f'TimeoutError: {e}')
            except Exception as e:
                print(f'\nan exception raised : {e}\n{traceback.format_exc()}')
                continue

    # 保存剩余的结果
    js=clean_surrogates(js)
    savejsonl(save_path,js)
 

if __name__=='__main__': 
     
    generate_system_profile_given_gpts(data_source="frgpts",advance_model_name='gpt-4o-mini',language='zh') 