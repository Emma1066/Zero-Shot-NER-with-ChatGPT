import os
import json
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import traceback
from openai.embeddings_utils import cosine_similarity
import tiktoken
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

from os import path
import sys
# 导入parent目录下的模块
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import max_tokens, num_tokens_from_messages, load_data, save_data, assert_gpt35_turbo_16k
from const import dataset_language_map


def load_demo_data(path, demo_num):
        if demo_num:
            demo_data = load_data(path)
        else:
            demo_data = list()

        return demo_data

class PromptGenerator(object):
    def __init__(self, args) -> None:
        self.args = args
        self.dataname = args.dataname
        self.lanuage = dataset_language_map[self.dataname]
        prompt_pool_choices = {
            "en": PromptPoolEnglish,
            "zh": PromptPoolChinese
        }
        self.prompt_pool = prompt_pool_choices[self.lanuage](args.dataname)

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

    def retrieval_demo(self, query_sample, demo_data, query_emb=None, demo_embs=None):
        '''给query_sample检索适配的demo'''
        if self.args.few_shot_setting == "zs":
            return []
        if self.args.few_shot_setting == "fixed":
            return demo_data
        if self.args.demo_retrieval_method == "random":
            return self.retrieval_demo_by_random(demo_data, self.args.few_shot_number)
        elif self.args.demo_retrieval_method in ["GPTEmbCos"]:
            return self.retrieval_demo_by_emb(demo_data, self.args.few_shot_number, query_emb, demo_embs)    
        else:
            raise ValueError(f"Wrong demo_retrieval_method={self.args.demo_retrieval_method}")

    def generate_prompt_per_query(
            self, 
            query_sample, 
            demo_data,
            query_emb=None,
            demo_embs=None
    ):
        '''生成单个query的prompt'''
        demos = self.retrieval_demo(query_sample, demo_data, query_emb=query_emb, demo_embs=demo_embs)
        prefix = self.prompt_pool.get_prompt_prefix(self.args)
        demos_prompts = []
        tmp_prompt = prefix
        exceed_max_len_flag = False
        for _, demo in enumerate(demos):
            demo_prompt = self.prompt_pool.get_prompt_for_demo(self.args, demo)
            
            # 限制最大长度
            tmp_prompt += demo_prompt
            if num_tokens_from_messages(tmp_prompt) > max_tokens(self.args.model) - args.output_token_len:
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

    def generate_prompt_batch(
            self, 
            query_data, 
            demo_data,
            query_embs=None,
            demo_embs=None
    ):
        '''生成整个数据集prompt'''
        data_prompts = []
        exceed_max_len_cnt = 0
        for i_query, query in enumerate(tqdm(query_data, desc="generate prompt")):
            query_emb = None
            if self.args.demo_retrieval_method in ["GPTEmbCos"] and self.args.few_shot_setting in ["full","pool"]:
                assert query_embs is not None
                query_emb = query_embs[i_query]
            prompt, exceed_max_len_flag = self.generate_prompt_per_query(
                query, 
                demo_data,
                query_emb=query_emb,
                demo_embs=demo_embs
            )

            # 限制最大长度
            '''
            if num_tokens_from_messages(prompt) > max_tokens(self.args.model) - 1000:
                print("\n超出长度限制:\nidx = {}, sentence = \n{}\nprompt = \n{}".format(query["idx"], query["sentence"], prompt))
                prompt = prompt[:max_tokens(self.args.model) - 500]       
                print("剪裁后:\nprompt = \n{}".format(prompt))
            '''

            if exceed_max_len_flag:
                exceed_max_len_cnt += 1
                
            query_prompt = {
                "idx": query["idx"],
                "sentence": query["sentence"],
                "label": query["label"],
                "prompt": prompt
            }
            data_prompts.append(query_prompt)
        
        print(f"\n超出长度限制样本数 = {exceed_max_len_cnt}")

        return data_prompts

def main(args):
    # 加载数据
    query_data = load_data(args.query_data_path)
    demo_data = load_demo_data(args.demo_data_path, demo_num=args.few_shot_number)

    if args.few_shot_setting == "fixed":
        assert len(demo_data) == args.few_shot_number

    # 加载embedding
    if args.demo_retrieval_method in ["GPTEmbCos"] and args.few_shot_setting in ["pool", "full"]:
        query_embs = np.load(args.query_embs_path)
        demo_embs = np.load(args.demo_embs_path)

        # 如果demo不是full training set，则取demo set对应的emb set
        # if args.few_shot_setting == "pool":
        #     demo_embs = fetch_indexed_embs(demo_embs, demo_data)

        assert len(query_data) == len(query_embs)
        assert len(demo_data) == len(demo_embs)
    else:
        query_embs = None
        demo_embs = None
    
    # 生成prompt
    prompt_generator = PromptGenerator(args=args)
    prompts = prompt_generator.generate_prompt_batch(
        query_data, 
        demo_data,
        query_embs=query_embs,
        demo_embs=demo_embs
    )

    save_data(args.save_prompt_path, prompts)


def get_paths(args):
    dataname = args.dataname
    if args.dataname == "ace04en":
        dataname = f"{args.dataname}/{args.folder}"

    # 标签集路径 + 加载
    args.abb2labelname_path = f"OPENAI/data/{args.task}/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # 标记parse信息来源
    parse_postfix = args.parse_tool
    
    # 选择embedding
    # emb = None
    # if args.demo_retrieval_method in ["SBertEmbCos", "GPTEmbCos"]:
    #     emb_choices = {"SBertEmbCos": "SBERTEmb", "GPTEmbCos": "GPTEmb"}
    #     emb = emb_choices[args.demo_retrieval_method]
    if args.demo_retrieval_method:
        if "SBertEmb" in args.demo_retrieval_method:
            emb = "SBertEmb"
        elif "GPTEmb" in args.demo_retrieval_method:
            emb = "GPTEmb"
        else:
            emb = None
    elif args.demo_select_method:
        if "GPTEmb" in args.demo_select_method:
            emb = "GPTEmb"
        elif "SBert" in args.demo_select_method:
            emb = "SBert"
        else:
            emb = None
    else:
        emb = None
    
    # query数据路径
    datamode = args.datamode
    # if args.dataname == "conll2003":
    #     datamode = "conllpp_test"
    args.query_data_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}.json"
    if args.tool_aug:
        args.query_data_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}_parse_{parse_postfix}.json"
    args.query_embs_path = f"OPENAI/data/{args.task}/{dataname}/{datamode}_{emb}.npy"
    
    # demo数据路径
    if args.few_shot_setting == "zs":
        args.demo_data_path = None
    elif args.few_shot_setting == "full":
        demo_filename = "train.json"
        if args.tool_aug:
            if "Pos" in args.tool_aug or "Dep" in args.tool_aug or "Con" in args.tool_aug or "Tok" in args.tool_aug:
                demo_filename = f"train_parse_{parse_postfix}.json"
        elif args.reason_hint:
            if "pos" in args.reason_hint or "dep" in args.reason_hint or "con" in args.reason_hint or "tok" in args.reason_hint:
                demo_filename = f"train_parse_{parse_postfix}.json"
        args.demo_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_filename}"
        args.demo_embs_path = f"OPENAI/data/{args.task}/{dataname}/train_{emb}.npy"          

    elif args.few_shot_setting in ["fixed", "pool"]:
        demo_folder = f"demo_{args.few_shot_setting}"
        demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}.json"
        if args.tool_aug:
            if "Pos" in args.tool_aug or "Dep" in args.tool_aug or "Con" in args.tool_aug or "Tok" in args.tool_aug:
                demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"    
        elif args.reason_hint:
            if "pos" in args.reason_hint or "dep" in args.reason_hint or "con" in args.reason_hint or "tok" in args.reason_hint:
                demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"
        

        args.demo_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_filename}"

        demo_embs_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_{emb}.npy"
        args.demo_embs_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_embs_filename}"          

    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")
  
    # prompt存储路径
    prompt_folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        prompt_folder = f"fs_{prompt_folder}"
    if args.few_shot_setting in ["fixed", "pool"]:
        prompt_folder = f"{prompt_folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        prompt_folder = f"{prompt_folder}_{args.demo_retrieval_method}"
    if args.tool_aug:
        prompt_folder = f"{prompt_folder}_tool"
    
    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"

    parse_tool_folder = args.parse_tool
    prompt_dir = f"OPENAI/prompts/{args.task}/{parse_tool_folder}/{dataname}/{prompt_folder}"
    # if args.parse_tool != "hanlp":
    #     prompt_dir = f"OPENAI/prompts/{args.task}/{args.parse_tool}/{dataname}/{prompt_folder}"

    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    args.save_prompt_path = os.path.join(prompt_dir, prompt_filename)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--folder", default=0, type=int) # only for ace04
    parser.add_argument("--datamode", default="test", type=str)
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--output_token_len", default=1000, type=int)
    
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    # [None, key_noun, key_noun_verb, key_noun_verb_con_dep, key_con_dep_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=300, type=int)
    # choices=["random_42", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=["random", "GPTEmbCos", "SBERTEmbCos"])
    parser.add_argument("--few_shot_number", default=1, type=int)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # 解析工具选择
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])
    
    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]

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

    # 根据最大上下文长度需求，更改模型
    # 切换为16k模型
    assert_gpt35_turbo_16k(args, chat_paradigm="standard")

    args = get_paths(args)

    print("---------- Generate prompts ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

