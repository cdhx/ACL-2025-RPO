import re
import subprocess
import os
import socket
import threading
import requests
import time
from contextlib import closing
from pathlib import Path
import json

import threading
 

def request_local(prompt,ls,temperature=0,n=1):
    if temperature==0 and n!=1:
        raise ValueError('best_of must be 1 when using greedy sampling. if n>1, make sure temperature>0 to avoid greedy sampling')
    # 必须传入ls对象, 使用的是chat的方式
    if type(prompt)==list and 'role' in prompt[0].keys() and 'content' in prompt[0].keys():
        messages=prompt
    if type(prompt)==str:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ] 

    from openai import OpenAI 

    openai_api_base = ls.end_point

    openai_api_key = "EMPTY"
    client = OpenAI(    
        api_key=openai_api_key,
        base_url=openai_api_base,
    )  
    retry_times=0
    while True:
        chat_response = client.chat.completions.create(
            model=ls.served_name,
            messages=messages,
            temperature=temperature,   
            n=n,     
        )
        # 经常抽风输出\\n\\n\\n
        if n==1:
            if chat_response.choices[0].message.content.replace('\\n','')!='':    
                return chat_response.choices[0].message.content 
        else:
            result=[x.message.content.replace('\\n','')  for x in chat_response.choices]
            result = [x for x in result if x!='']
            if len(result)!=0:
                return result
        retry_times+=1
        if retry_times>100:
            raise ValueError("ERROR WHEN CALLLING LOCAL MODEL")
            return 'ERROR WHEN CALLLING LOCAL MODEL'
 

def parse_model_name(model_name_or_path):
    if not model_name_or_path:
        return None
    name = Path(model_name_or_path).name
    model_dir = Path(model_name_or_path).parent
    if name.startswith("checkpoint-"):
        name = Path(model_dir.name) / name
    return str(name)

class LocalServer(threading.Thread):
    condition_met = threading.Event()
    STARTED_INFO = "Uvicorn running on http://0.0.0.0"

    def __init__(self, ckpt_path, tp, gpus, port, chat_template_path=None):
        self.served_name = parse_model_name(ckpt_path)
        self.ckpt_path = ckpt_path
        self.tp = str(tp)
        self.stdout = None
        self.stderr = None
        self.process = None
        self.port = port if port else 8000
        self.end_point = "http://127.0.0.1:{}/v1".format(port)
        self.gpus = gpus

        self.chat_template_path = chat_template_path   
        threading.Thread.__init__(self)
        self.setDaemon(True)

    def run(self):
        # default end point
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus
        # check alive vllm server
        cmd = [
            'python', '-m', "vllm.entrypoints.openai.api_server",
            "--model", self.ckpt_path,
            "--served-model-name", self.served_name,
            "--tensor-parallel-size", self.tp,
            "--port", str(self.port)
        ]
        if self.chat_template_path:
            cmd.extend(["--chat-template", self.chat_template_path])

        # Check started servers
        cmd_str = " ".join(cmd)
        process = subprocess.Popen("ps -wwf | grep vllm.entrypoints | grep -v grep", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        output, _  = process.communicate()
        print("\n================= DETECTED SERVICE ======================\n{}\n".format(output))
        # start current vllm if not alive
        if cmd_str not in output:
            lines = output.split("\n")
            # kill all vllm server first
            lines = list(filter(lambda x: "--port {}".format(self.port) in x, lines))
            if lines:
                print("------------------ WILL BE KILLED ------------------------\n{}")
                print("[KILL] {}".format(lines[0]))
                pid = re.split(" +", lines[0])[1]
                print(pid)
                p = subprocess.Popen(["kill", "-9", pid])
                p.communicate()

            # start current
            print("==================== START NEW SERVER on GPU: {}, Port: {} ========================\n{}\n".format(self.gpus, self.port, " ".join(cmd)))
            self.process = subprocess.Popen(cmd, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, preexec_fn=os.setsid)

            out = ""
            while LocalServer.STARTED_INFO not in out:
                out = self.process.stdout.readline()
                print(out.strip())
                if self.process.poll() is not None:
                    break
            LocalServer.condition_met.set()
            stdout, stderr = self.process.communicate()

    def wait(self):
        LocalServer.condition_met.wait()


def get_gpu_index_for_sufficient_memory(free_mem_size_mib):
    import pynvml
    pynvml.nvmlInit()
    gpus = []
    gpu_num = pynvml.nvmlDeviceGetCount()
    for i in range(0, gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free // 1024 ** 2
        if free_mem > free_mem_size_mib:
            gpus.append(i)
    return gpus
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def prepare_local_service(ckpt):
    # if "72B" in ckpt or "72b" in ckpt:
    #     tp = 8 

    LEAST_MEM_MB = 70000
    available_gpus = get_gpu_index_for_sufficient_memory(LEAST_MEM_MB) 
    served_name = parse_model_name(ckpt)

    gpus = ",".join([str(_) for _ in available_gpus]) 
    port = find_free_port() 
 
    # 获取jinja信息
    with open(os.path.join(ckpt,'tokenizer_config.json'), 'r', encoding='utf-8') as file:
        config_file_content = json.load(file)

    content=config_file_content['chat_template'].replace('\n','\\n')
    
    chat_template_abs_path=os.path.join('../../data/jinja',served_name+'.jinja')
    # 写入
    with open(chat_template_abs_path, 'w', encoding='utf-8') as file:
        file.write(content) 
    tp = len(available_gpus)
    print('Use jinja file: ', chat_template_abs_path)
    ls = LocalServer(ckpt, tp, gpus, port, chat_template_abs_path)
    
    ls.start()
    ls.wait()
    return ls


if __name__ == "__main__":
    pass
 