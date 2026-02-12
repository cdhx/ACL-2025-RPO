

# 生成一些config file


import argparse

import json
import os

import yaml
import argparse

from transformers import AutoTokenizer

from tqdm import tqdm



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
    with open(temp_file, 'w', encoding='utf-8') as fp:
        json.dump(json_info, fp, indent=indent, ensure_ascii=False)
    # 使用原子操作重命名文件
    os.replace(temp_file, final_file)  # 在大多数操作系统上，这是原子操作 




def set_state(config):
    eval_dataset=config['eval_dataset']
    eval_model=config['eval_model']
    model=config['model']

    stage='dpo'
    finetuning_type='lora'
    
    gpu_num=config['gpu_num']
    gradient_accumulation_steps=config['gradient_accumulation_steps']
    epoch=config['epoch']
    max_examples=config['max_examples']
    learning_rate=config['learning_rate']
    flash_atten=config['flash_atten']
    cutoff_len=config.get('cutoff_len')
    data_size=config.get('data_size')
    comment=config['comment'] 
 
    # 模型路径是否存在
    if os.path.exists(model):
        model_path=model 
    else:        
        raise ValueError("model path "+model+" not exist")
        sys.exit(1)     

    # 是否开多卡deepspeed
    if gpu_num>1:
        deepspeed=True
        offload=True
    else:
        deepspeed=False
        offload=False
    
   
    assert gpu_num<=8, "gpu_num greater than 8 is not implemented" 

    
    # 模型名
    model_name=model_path.split('/')[-1]  
    dataset_name=config['dataset_name']
    # 实验名
    exp_name=model_name+'_'+stage+'_'+dataset_name+'_lr_'+str(learning_rate)+('_'+str(int(max_examples))+'_data' if max_examples<1e5 else '')+('_'+comment if comment != '' else '')

 
    # template
    config_file_content=readjson(os.path.join(model_path,'config.json'))
    template=config_file_content['model_type']
    # 模板是llama3和qwen
    template = 'llama3' if template  == 'llama' else 'qwen' if template == 'qwen2' else template


    # 训练的yaml,用dpo的
    file_name = "/mnt/workspace/hx/LLaMA-Factory/examples/train_lora/llama3_lora_dpo.yaml"
    # 读取yaml样例
    with open(file_name) as f:
        doc = yaml.safe_load(f) 

    # 处理deepspeed和offload
    if deepspeed:
        doc['deepspeed'] = 'examples/deepspeed/ds_z3_config.json'
        if offload:
            doc['deepspeed'] =  'examples/deepspeed/ds_z3_offload_config.json'
    else:
        if 'deepspeed' in doc:
            del doc['deepspeed']
 
    if cutoff_len is None or data_size is None:
        # 自动调整step记录打印保存相关的参数 
        data_info=readjson('/mnt/workspace/hx/LLaMA-Factory/data/dataset_info.json')

        # 看之前存没存过
        cached=False
        if 'num_samples' in data_info[dataset_name].keys():
            if 'qwen' in template and 'qwen_max_tokens' in data_info[dataset_name].keys() or 'llama' in  template and 'llama_max_tokens' in data_info[dataset_name].keys():
                cached=True
        # 之前存过
        if cached:
            data_size=data_info[dataset_name]['num_samples']
            if 'qwen' in template:
                cutoff_len=data_info[dataset_name]['qwen_max_tokens']                
            if 'llama' in template:
                cutoff_len=data_info[dataset_name]['llama_max_tokens']
        # 之前没存过
        else:
            # 获取两个路径的拼接
            data_path=os.path.join('/mnt/workspace/hx/LLaMA-Factory/data/',data_info[dataset_name]['file_name'])
            with open(data_path, 'r', encoding='utf-8') as file:
                data_js=json.load(file)
                
            # 根据数据量，epoch和batch size确定step相关
            data_size=len(data_js)
    
            tokenizer = AutoTokenizer.from_pretrained(model_path)  
    
            js_token_len=[]
 
            for item in tqdm(data_js):
                history=[]
                for turn in item['conversations']+[item['chosen']]:
                    from2role={'human':'user','system':'system','gpt':'assistant'}
                    role='user' if turn['from']=='human'  else 'user'
                    history.append({'role':from2role[turn['from']],'content':turn['value']})
                js_token_len.append(len(tokenizer.apply_chat_template(history, tokenize=True)))
 
            js_token_len.sort()
            len_percent_95 = js_token_len[int(len(js_token_len)*0.95)]# 长度95%分位点
            # 太长的警告一下
            if len_percent_95>8192:
                raise ValueError("data is too long")
            avaliable_max_len=[256,512,1024,2048,4096,8192]
            cutoff_len = min([x for x in avaliable_max_len if x>len_percent_95])
            print('***********************')
            print('data size:',data_size)
            print('95 percent of data less than ',len_percent_95,' tokens. choose ',cutoff_len,' as cutoff len.')
            print('model_path:',model_path)            
            print('***********************\n\n')

            data_info[dataset_name]['num_samples']=data_size
            if 'qwen' in template:
                data_info[dataset_name]['qwen_max_tokens']=cutoff_len
            elif 'llama' in template:
                data_info[dataset_name]['llama_max_tokens']=cutoff_len
            else:
                raise ValueError('model is not llama or qwen')
            savejson('/mnt/workspace/hx/LLaMA-Factory/data/dataset_info.json',data_info)
 
    mini_batch=max(1,int(8192/cutoff_len))  
    # qwen 7b 28头28层3584 hidden，llama 8b 32头32层4096 hidden
    # qwen 72b 64头80层8192，llama 70b 64头80层8192 hidden
    # 70b的mini_batch要再除2才行,这个if模糊判断一下是不是70b的模型 
    if config_file_content['num_attention_heads']>32 and config_file_content['hidden_size']>4096:    
        mini_batch=mini_batch//2 


    global_batch_size=gpu_num*gradient_accumulation_steps*mini_batch
    data_size=min(data_size,max_examples) # 数据量
    global_step=data_size//global_batch_size*epoch # global step
    logging_steps=max(1,global_step//100) # 打印100次或者1个step打印一次
    save_steps=logging_steps*5 # 保存的step是打印step的5倍即可
        

    # 基础参数
 
    doc['model_name_or_path'] = model_path 
    doc['dataset'] = dataset_name
    doc['template'] = template
    doc['cutoff_len'] = cutoff_len
    doc['max_samples'] = max_examples
    model_save_path='/mnt/workspace/hx/LLaMA-Factory/saves/'+model_name+'/lora/'+exp_name 
 
    doc['output_dir'] = model_save_path
    doc['logging_steps'] = logging_steps
    doc['save_steps'] = save_steps
    doc['eval_steps'] = save_steps
    doc['gradient_accumulation_steps'] = gradient_accumulation_steps
    doc['overwrite_cache']=config['overwrite']
    doc['overwrite_output_dir']=config['overwrite']
    
    doc['report_to'] = 'tensorboard'
    doc['logging_dir'] = os.path.join(model_save_path,'runs')
    if flash_atten:
        doc['flash_attn'] = 'fa2'
        
     
    doc['per_device_train_batch_size'] = mini_batch
    doc['per_device_eval_batch_size'] = mini_batch 
 
 
    if learning_rate is not None:
        doc['learning_rate'] = config['learning_rate'] 
    
    # 保存训练的yaml
    train_yaml_save_path = '/mnt/workspace/hx/LLaMA-Factory/examples/train_'+finetuning_type+'/'+model_name+'/'+exp_name+'.yaml'
    directory = os.path.dirname(train_yaml_save_path)
    # 如果目录不存在，则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(train_yaml_save_path, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

    # merge lora的yaml,sft和dpo都一样
 
    file_name = "/mnt/workspace/hx/LLaMA-Factory/examples/merge_lora/llama3_lora_sft.yaml"
    with open(file_name) as f:
        doc = yaml.safe_load(f) 

    doc['model_name_or_path'] = model_path
    doc['adapter_name_or_path'] = model_save_path
    doc['template'] = template 
    doc['export_dir'] = '/mnt/workspace/hx/LLaMA-Factory/models/'+model_name+'/lora/'+exp_name

    export_yaml_save_path = '/mnt/workspace/hx/LLaMA-Factory/examples/merge_lora/'+model_name+'/'+exp_name+'.yaml'
    directory = os.path.dirname(export_yaml_save_path)
    # 如果目录不存在，则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(export_yaml_save_path, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

    # 先不搞多机了，也没有模型并行的能力
    # 处理多卡的情况 
    if gpu_num%8!=0 and gpu_num>8:
        raise ValueError("gpu num shoulf be less than 8 or multiple of 8")

    llama_factoty_train_bash=("FORCE_TORCHRUN=1 " if deepspeed else "")+ "llamafactory-cli train "+train_yaml_save_path        

    llama_factoty_export_bash="\nllamafactory-cli export "+export_yaml_save_path+' \necho "===============\nFINISED LORA MERGE...\n===============\n"' if finetuning_type=='lora' else ""
    
    bash_comand=f"""
  

conda activate llama_factory
cd /mnt/workspace/hx/LLaMA-Factory  
{llama_factoty_train_bash} 
echo "===============\nTraining Finished...\n===============\n"
{llama_factoty_export_bash} 
echo "===============\nMerge lora Finished...\n===============\n"

 """
 
    return bash_comand

 


if __name__ == "__main__":
 
    config={    
        'finetuning_type':'lora',
        'dataset_name':'rdpo_js',
        # 'gpu_num':4, # use 4 GPU for qwen 7b
        'gpu_num':8, # other use 8 gpu

        # 'mini_batch':4,
        'gradient_accumulation_steps':64,
        'epoch':3,
        'flash_atten':True,
        'learning_rate':5e-4,
        'max_examples':int(1e9),        
        # 'max_examples':5000,   
        'epoch':3,
        "overwrite":False,
        # "overwrite":True,
        
        'infer_concurrent': 32,
        'eval_concurrent': 20,
        # 'cutoff_len':2048,
        # 'data_size':8676, 
        'comment':'', 
    }

    # 70B模型dpo minibatch 2
    # 7B模型dpo 4卡 minibatch 4 token 2048 minibatch 2 token 4096 minibatch 1 token 8192
    # 7B模型sft 4卡 minibatch 4 token 4096 minibatch 1 token 16384
    # 7B模型kto 4卡 minibatch 8 token 2048

 

    model_list=[          
        # '/mnt/workspace/hx/Models/Llama-3.1-8B-Instruct',
        # '/mnt/workspace/hx/Models/Qwen2.5-7B-Instruct',         
 
        '/mnt/workspace/hx/Models/Llama-3.1-70B-Instruct',
        '/mnt/workspace/hx/Models/Qwen2.5-72B-Instruct',
          

        ]
    print("""
cd /mnt/workspace/hx
. anaconda3/bin/activate     
    """) 

    learning_rate_list=[1e-5] 
    for model in model_list:  
        config['model']=model 
        print(set_state(config))
    
    

