import json
import time
import logging, logging.config
import sys
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import openai
from openai.embeddings_utils import cosine_similarity
import tiktoken
import random
import copy

from os import path
import sys
# 导入parent目录下的模块
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) )) ) # 往上2层目录

from utils import get_logger, load_data, max_tokens, num_tokens_from_messages
from utils_parse_answer import response_2_prediction
from const import my_api_keys, dataset_language_map
from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

logger = logging.getLogger()

class SafeOpenai:
    def __init__(self, keys=None, start_id=None, proxy=None):
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys

        if start_id is None:
            start_id = random.sample(list(range(len(keys))), 1)[0]

        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        openai.proxy = proxy
        openai.api_key = self.key[self.key_id % len(self.key)]


    def chat(self, 
            model, 
            messages, 
            temperature, 
            stop=None,
            return_text=False, 
            sleep_seconds=2, 
            max_read_timeout=1, 
            max_http=1):
        timeout_cnt = 0
        http_cnt = 0
        while True:
            if timeout_cnt >= max_read_timeout:
                logger.info(f"timeout_cnt exceed max_read_timeout {max_read_timeout}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
            elif http_cnt >= max_http:
                logger.info(f"http_cnt exceed max_read_timeout {max_http}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."                            
            try:

                completion = openai.ChatCompletion.create(
                    model=model, 
                    messages=messages, 
                    temperature=temperature,
                    stop=stop,
                    timeout=3)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)                
            except openai.error.APIConnectionError as e:
                logger.info(f"Failed to connect to OpenAI API: {e}")
                if "HTTPSConnectionPool" in str(e) and 'Remote end closed connection without response' in str(e):
                    http_cnt += 1
                    logger.info(f"http_cnt + 1 = {http_cnt}")                
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)  
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                if "please check your plan and billing details" in str(e):
                    logger.info(f"delete key: {self.key[self.key_id]}, key id: {self.key_id}")
                    del self.key[self.key_id]
                    self.key_id -= 1                
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)  
            except Exception as e:
                logger.info(str(e))
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "Read timed out" in str(e):
                    timeout_cnt += 1
                    logger.info(f"timeout_cnt + 1 = {timeout_cnt}")


                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)                

        if return_text:
            completion = completion['choices'][0]['message']['content']
        return completion

    def text(self, *args, return_text=False, sleep_seconds=2, max_timeout=3, **kwargs):
        timeout_cnt = 0
        while True:
            if timeout_cnt >= max_timeout:
                logger.info(f"timeout_cnt exceed max_timeout {max_timeout}. Skip this request.")
                return f"Timeout exceeds {max_timeout}, skip."            
            try:
                completion = openai.Completion.create(*args, **kwargs)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)                
            except openai.error.APIConnectionError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)  
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)  
            except Exception as e:
                logger.info(str(e))
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "" in str(e):
                    logger.info("timeout_cnt + 1")
                    timeout_cnt += 1
                
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(sleep_seconds)                   

        if return_text:
            completion = completion['choices'][0]['text']
        return completion
    
def run_llm(
        messages, 
        openai_key=None, 
        model_name="gpt-3.5-turbo", 
        temperature=0,
        stop=None
):
    agent = SafeOpenai(openai_key)
    response = agent.chat(
        model=model_name, 
        messages=messages, 
        temperature=temperature, 
        stop=stop,
        return_text=True)
    return response


class AskChatGPT(object):
    def __init__(self, args) -> None:
        self.args = args
        self.dataname = args.dataname
        self.lanuage = dataset_language_map[self.dataname]
        prompt_pool_choices = {
            "en": PromptPoolEnglish,
            "zh": PromptPoolChinese
        }
        self.prompt_pool = prompt_pool_choices[self.lanuage](args.dataname)

    def generate_responses_per_query(self, query):
        messages = [
            {"role": "user", "content": query["prompt"]}
        ]
        response = run_llm(
            messages,
            openai_key=self.args.api_key,
            model_name=self.args.model,
            temperature=self.args.temperature,
            stop=self.args.stop
        )

        query_resp = {
            "idx": query["idx"],
            "sentence": query["sentence"],
            "label": query["label"]
        }
        # 添加tool aug
        if self.args.tool_aug == "ToolTokCoarse":
            query_resp["tok/coarse"] = query["tok/coarse"]
        if self.args.few_shot_setting == "zs":
            query_resp["prompt"] = query["prompt"]

        query_resp["response"] = response

        return query_resp


    def generate_responses_per_query_multiquery(self, query, query_times=5, temperature=1.0):
        messages = [
            {"role": "user", "content": query["prompt"]}
        ]

        query_resp = {
            "idx": query["idx"],
            "sentence": query["sentence"],
            "label": query["label"]
        }    
        if self.args.few_shot_setting == "zs":
            query_resp["prompt"] = query["prompt"]    

        for i_time in range(query_times):
            response = run_llm(
                messages,
                openai_key=self.args.api_key,
                model_name=self.args.model,
                temperature=temperature,
                stop=self.args.stop
            )

            query_resp[f"response_{i_time}"] = response

        return query_resp
    
    def retrieval_demo_by_emb(self, demo_data, n_demo, query_emb, demo_embs):
        demo_df = pd.DataFrame(columns=["embedding"])
        demo_df["embedding"] = list(demo_embs)
        # about 1.4s
        demo_df["similarity"] = demo_df.embedding.apply(lambda x: cosine_similarity(x, query_emb))
        
        cos_sims = demo_df["similarity"]
        sorted_idxes = np.argsort(cos_sims).tolist()
        sorted_idxes.reverse()
        
        demos_selected = []
        cnt = 0
        while len(demos_selected) < n_demo:
            # 不选不含实体标签的样本
            # if len(sorted_idxes[cnt]["label"]) == 0:
            #     continue
            demos_selected.append(demo_data[sorted_idxes[cnt]])
            cnt += 1        

        # 把越相似的放越后面（与query越接近）
        demos_selected.reverse()

        return demos_selected    

    def retrieval_demo_by_random(self, demo_data, n_demo):
        demos_selected = []
        demos_idx = []
        while len(demos_selected) < n_demo:
            tmp_idx = random.choice(range(len(demo_data)))
            # 样例不重复
            if tmp_idx in demos_idx:
                continue

            # 不选不含实体标签的样本
            # if len(demo_data[tmp_idx]["label"]) == 0:
            #     continue

            demos_selected.append(demo_data[tmp_idx])
            demos_idx.append(tmp_idx)

        return demos_selected     


    def select_demo(self, demo_valt, demo_embs, query_sample, query_emb):
        if self.args.memory_selection == "random":
            return self.retrieval_demo_by_random(demo_valt, self.args.memory_shot)
        if self.args.memory_selection in ["SBertEmbCos", "GPTEmbCos"]:
            return self.retrieval_demo_by_emb(demo_valt, self.args.memory_shot, query_emb, demo_embs)
        else:
            raise ValueError(f"Wrong memory_selection_method={self.args.memory_selection}")

    def generate_prompt_per_query(self, query_sample, demos):
        '''生成单个query的prompt'''
        prefix = self.prompt_pool.get_prompt_prefix(self.args)
        demos_prompts = []
        tmp_prompt = prefix
        exceed_max_len_flag = False
        for _, demo in enumerate(demos):
            demo_prompt = self.prompt_pool.get_prompt_for_demo(self.args, demo)
            
            # 限制最大长度
            tmp_prompt += demo_prompt
            if num_tokens_from_messages(tmp_prompt) > max_tokens(self.args.model) - 1000:
                print("\n超出长度限制:\nidx = {}, sentence = \n{}".format(query_sample["idx"], query_sample["sentence"]))
                exceed_max_len_flag = True
                break

            demos_prompts.append(demo_prompt)

        demos_prompts = "".join(demos_prompts)
        postfix = self.prompt_pool.get_prompt_postfix(self.args, query_sample)

        prompt = prefix + demos_prompts + postfix
        
        if exceed_max_len_flag:
            print(prompt)

        return prompt, exceed_max_len_flag


    def generate_responses_batch(self, data_prompts, query_embs):
        args = self.args
        bar = tqdm(data_prompts, ncols=100)
        start_idx = 0
        # 断点续接
        # if args.start_time:
        #     pre_res = load_data(args.response_path)
        #     if len(pre_res) > 0:
        #         start_idx = len(pre_res)
        query_resp_memory_all = []
        query_embs_ememory_all = []
        exceed_max_len_cnt = 0
        with open(args.response_path, "ab", buffering=0) as realtime_f:
            for i_query, query in enumerate(bar):
                bar.set_description("Query ChatGPT NER")
                # 如果不是从第一条数据开始
                if i_query < start_idx:
                    continue

                query_emb = query_embs[i_query] if "Emb" in args.memory_selection else None
                # 从test memory中选择demos
                if len(query_resp_memory_all) <= args.memory_shot:
                    demos = query_resp_memory_all
                else:
                    demos = self.select_demo(query_resp_memory_all, query_embs_ememory_all, query, query_emb)
                # 生成prompt
                query["prompt"], exceed_max_len_flag = self.generate_prompt_per_query(query, demos)

                if exceed_max_len_flag:
                    exceed_max_len_cnt += 1

                if not args.consistency:
                    query_resp = self.generate_responses_per_query(query)
                else:
                    query_resp = self.generate_responses_per_query_multiquery(query, query_times=args.query_times, temperature=args.temperature)
                
                realtime_f.write((str(query_resp)+"\n").encode("utf-8"))
                
                # 存储test sample进入test memory
                query_resp_mem = copy.deepcopy(query_resp)
                # 将response转换成对应的预测答案
                query_resp_mem["label"] = response_2_prediction(args, query, query_resp_mem["response"])
                query_resp_memory_all.append(query_resp_mem)
                query_embs_ememory_all.append(query_emb)
                
        
        logger.info("Finished!")
        logger.info(f"超出长度限制样本数 = {exceed_max_len_cnt}")
        logger.info(f"response saved to: {args.response_path}")
        logger.info(f"used api_key: {args.api_key}")
        

def main(args):
    # 加载数据
    query_data = load_data(args.query_data_path)    

    # 加载embedding
    if args.memory_selection in ["SBertEmbCos", "GPTEmbCos"]:
        query_embs = np.load(args.query_embs_path)
        assert len(query_data) == len(query_embs)
    else:
        query_embs = None
    
    chatgpt_asker = AskChatGPT(args)

    # 获取ChatGPT回答
    chatgpt_asker.generate_responses_batch(
        query_data,
        query_embs
    )


def get_paths(args):
    dataname = args.dataname
    if args.dataname == "ace04en":
        dataname = f"{args.dataname}/{args.folder}"

    # 标签集路径 + 加载
    args.abb2labelname_path = f"OPENAI/data/{args.task}/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())        

    # query数据路径
    parse_postfix = "hanlp"
    datamode = args.datamode
    if args.dataname == "conll2003":
        datamode = "conllpp_test"
    args.query_data_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}.json"
    if args.tool_aug:
        args.query_data_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}_parse_{parse_postfix}.json"
    # 选择embedding
    if args.memory_selection in ["SBertEmbCos", "GPTEmbCos"]:
        emb_choices = {"SBertEmbCos": "SBERTEmb", "GPTEmbCos": "GPTEmb"}
        emb = emb_choices[args.memory_selection]
        args.query_embs_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}_{emb}.npy"

    # prompt存储路径
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder = f"fs_{folder}"
    if args.few_shot_setting in ["fixed", "pool"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder = f"{folder}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder = f"{folder}_tool"        

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = time.strftime("%m%d%H%M")
    if args.start_time:
        start_time = args.start_time
    datamode = args.datamode
    if args.dataname == "conll2003":
        datamode = "conllpp_test"        
    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_AskGPT.log"

    args.prompt_path = f"OPENAI/prompts/{args.task}/{dataname}/{folder}/{prompt_filename}"

    folder_resp = folder
    # test memory标记
    folder_resp = f"{folder_resp}_testmem_{args.memory_selection}_{args.memory_shot}"
    if args.consistency:
        folder_resp = f"{folder}_consist_{args.temperature}_{args.query_times}"
    response_dir = f"OPENAI/result/{args.task}/{dataname}/{folder_resp}"
    if not os.path.exists(response_dir):
        os.makedirs(response_dir)    
    args.response_path = os.path.join(response_dir, response_filename)

    # Logger setting
    folder_log = folder
    if args.consistency:
        folder_log = f"{folder}_consist_{args.temperature}_{args.query_times}"    
    log_dir = f"OPENAI/log/{args.task}/{dataname}/{folder_log}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"OPENAI/config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train, test"])
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--max_token_len", default=4096, type=int)
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=1, type=int)
    parser.add_argument("--demo_select_method", default="manual1", choices=["random", "GPTEmbClusterKmeans"])
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos"])
    parser.add_argument("--few_shot_number", default=5, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    
    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # test memory
    parser.add_argument("--memory_selection", default="SBertEmbCos", type=str, choices=["random", "SBertEmbCos", "GPTEmbCos"])
    parser.add_argument("--memory_shot", default=1, type=int)

    # 实验断点续接
    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()

    # 设置stop列表
    stop_ls = ["\n", "[]", "[{}]"]
    args.stop = stop_ls    
    
    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
        args.demo_retrieval_method = None
    if args.few_shot_setting == "zs":
        args.few_shot_number = 0 
        args.demo_retrieval_method = None

    if args.reason_hint is None:
        args.reason_hint_pos = None
        args.reason_hint_person = None

    if args.tool_aug is None:
        args.tool_desc = None        

    # 如果用consistency，则温度需大于零
    if not args.consistency:
        args.temperature = 0

    if args.consistency:
        assert args.temperature > 0
        assert args.query_times > 0
    else:
        assert args.temperature == 0


    args = get_paths(args)


    args.api_key = my_api_keys

    logger.info("---------- Ask ChatGPT ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)