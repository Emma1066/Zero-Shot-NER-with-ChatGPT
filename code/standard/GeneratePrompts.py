import os
import json
from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

import sys
sys.path.append("code")
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

    def retrieval_demo(self, query_sample, demo_data):
        if self.args.few_shot_setting == "zs":
            return []
        if self.args.few_shot_setting == "fixed":
            return demo_data

    def generate_prompt_per_query(
            self, 
            query_sample, 
            demo_data,
    ):
        demos = self.retrieval_demo(query_sample, demo_data)
        prefix = self.prompt_pool.get_prompt_prefix(self.args)
        demos_prompts = []
        tmp_prompt = prefix
        exceed_max_len_flag = False
        for _, demo in enumerate(demos):
            demo_prompt = self.prompt_pool.get_prompt_for_demo(self.args, demo)
            
            # max len constraint
            tmp_prompt += demo_prompt
            if num_tokens_from_messages(tmp_prompt) > max_tokens(self.args.model) - args.output_token_len:
                print("\nExceed max len:\nidx = {}, sentence = \n{}".format(query_sample["idx"], query_sample["sentence"]))
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
            demo_data
    ):
        data_prompts = []
        exceed_max_len_cnt = 0
        for i_query, query in enumerate(tqdm(query_data, desc="generate prompt")):
            prompt, exceed_max_len_flag = self.generate_prompt_per_query(
                query, 
                demo_data
            )

            if exceed_max_len_flag:
                exceed_max_len_cnt += 1
                
            query_prompt = {
                "idx": query["idx"],
                "sentence": query["sentence"],
                "label": query["label"],
                "prompt": prompt
            }
            data_prompts.append(query_prompt)
        
        print(f"\n# Exceeding max len = {exceed_max_len_cnt}")

        return data_prompts

def main(args):
    # load data
    query_data = load_data(args.query_data_path)
    demo_data = load_demo_data(args.demo_data_path, demo_num=args.few_shot_number)

    if args.few_shot_setting == "fixed":
        assert len(demo_data) == args.few_shot_number

    # generate prompt
    prompt_generator = PromptGenerator(args=args)
    prompts = prompt_generator.generate_prompt_batch(
        query_data, 
        demo_data
    )

    save_data(args.save_prompt_path, prompts)

def get_paths(args):
    dataname = args.dataname

    # label set
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    parse_postfix = args.parse_tool
    
    datamode = args.datamode

    args.query_data_path = f"data/{dataname}/{datamode}.json"
    if args.tool_aug:
        args.query_data_path = f"data/{dataname}/{datamode}_parse_{parse_postfix}.json"
    
    # demo data path
    if args.few_shot_setting == "zs":
        args.demo_data_path = None

    elif args.few_shot_setting in ["fixed"]:
        demo_folder = f"demo_{args.few_shot_setting}"
        demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}.json"
        if args.tool_aug:
            if "pos" in args.tool_aug or "Dep" in args.tool_aug or "Con" in args.tool_aug or "Tok" in args.tool_aug:
                demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"    
        elif args.reason_hint:
            if "pos" in args.reason_hint or "dep" in args.reason_hint or "con" in args.reason_hint or "tok" in args.reason_hint:
                demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"

        args.demo_data_path = f"data/{dataname}/{demo_folder}/{demo_filename}"      
    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")
  
    # prompt saving path
    prompt_folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed"]:
        prompt_folder = f"fs_{prompt_folder}"
    if args.few_shot_setting in ["fixed"]:
        prompt_folder = f"{prompt_folder}_{args.demo_select_method}_{args.demo_size}"
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
    prompt_dir = f"prompts/{parse_tool_folder}/{dataname}/{prompt_folder}"

    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    args.save_prompt_path = os.path.join(prompt_dir, prompt_filename)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--output_token_len", default=1000, type=int)
    
    # prompt
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "zs"])
    parser.add_argument("--demo_size", default=3, type=int)
    parser.add_argument("--demo_select_method", default="random_42")
    parser.add_argument("--few_shot_number", default=3, type=int)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # parsing tool
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])
    
    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]

    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
    if args.few_shot_setting == "zs":
        args.few_shot_number = 0        
    if args.reason_hint is None:
        args.reason_hint_pos = None
        args.reason_hint_person = None
    if args.tool_aug is None:
        args.tool_desc = None

    assert_gpt35_turbo_16k(args, chat_paradigm="standard")

    args = get_paths(args)

    print("---------- Generate prompts ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

