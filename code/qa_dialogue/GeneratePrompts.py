import os
import json
from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

import sys
sys.path.append("code")
from utils import max_tokens, num_tokens_from_messages, load_data, save_data, assert_gpt35_turbo_16k
from const import dataset_language_map, dataset_label_order_map


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

    def generate_prompt_per_query(
            self, 
            query_sample, 
            label_order,
            demo_data
    ):
        tot_prompt = ""
        exceed_max_len_flag = False
        prompt = {}
        # task description
        task_desc =  self.prompt_pool.get_task_desc(self.args)
        prompt["task_description"] = task_desc
        tot_prompt += task_desc

        # output format constraint
        prompt["output_constraint"] = self.prompt_pool.get_output_constraint(form=args.output_format)
        if args.output_constraint_pos == "behindT":
            tot_prompt += prompt["output_constraint"]

        # answer and question hint
        prompt["answer_hint"] = self.prompt_pool.get_answer_hint()
        prompt["question_hint"] = self.prompt_pool.get_question_hint()

        # syntactic prompting
        prompt["reason_hint"] = self.prompt_pool.get_reason_hint(args)

        # demonstrations (future work)
        prompt["demos_prompts"] = ""

        # query information
        query_info =  self.prompt_pool.get_query_info(self.args, query_sample)
        prompt["query_information"] = query_info
        tot_prompt += query_info           
        # each question for the query sample
        question_ls =[]
        for _, target_types in enumerate(label_order):
            tmp_question = self.prompt_pool.get_question(self.args, target_types)
            question_ls.append(tmp_question)
            tot_prompt += tmp_question
            if args.output_constraint_pos == "behindQ":
                tot_prompt += prompt["output_constraint"]
        prompt["questions"] = question_ls

        if num_tokens_from_messages(tot_prompt) > max_tokens(self.args.model) - args.output_token_len:
            print("\nExceed max len:\nidx = {}\nsentence = \n{}".format(query_sample["idx"], query_sample["sentence"]))    
            exceed_max_len_flag = True

        if exceed_max_len_flag:
            print(prompt)

        return prompt, exceed_max_len_flag

    def generate_prompt_batch(
            self, 
            query_data, 
            demo_data,
    ):
        data_prompts = []
        exceed_max_len_cnt = 0
        label_order = self.args.label_order
        
        for i_query, query in enumerate(tqdm(query_data, desc="generate prompt")):
            prompt, exceed_max_len_flag = self.generate_prompt_per_query(
                query, 
                label_order,
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

        print(f"\n# (samples exceeding max len) = {exceed_max_len_cnt}")

        return data_prompts

def main(args):
    # Load data
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
    
    if args.few_shot_setting == "zs":
        args.demo_data_path = None
    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")

    # prompt saving path
    prompt_folder = f"{args.few_shot_setting}"
    if args.tool_aug:
        prompt_folder = f"{prompt_folder}_tool"    
    # add qa-tag in path
    if args.demo_form == "dialogue":
        qa_tag = "classDia"
        if args.output_constraint_pos == "behindT":
            qa_tag += "_BT"
    else:
        raise ValueError(f"Unrecognized demo_form = {args.demo_form}")          
    prompt_folder = f"{qa_tag}_{args.order}_{prompt_folder}" 
    
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
    # Data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
    # Model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--output_token_len", default=1000, type=int)
    
    # prompt
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="second", choices=[None, "first", "second"])
    parser.add_argument("--reason_hint_pos", default="f", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["zs"])
    parser.add_argument("--demo_size", default=0, type=int)
    parser.add_argument("--few_shot_number", default=0, type=int)
    parser.add_argument("--demo_form", default="dialogue", type=str, choices=["dialogue"])
    # output format
    parser.add_argument("--output_format", default="json", type=str, choices=["json", "list"])
    parser.add_argument("--output_constraint_pos", default="behindQ", type=str, choices=["behindT", "behindQ"], help="behindT: behind task description. behindQ: behind question.")

    # QA setting    
    parser.add_argument("--order", default=None)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # parsing tool
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])
    
    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]
    args.label_order = dataset_label_order_map[args.dataname][args.order]

    if args.few_shot_setting == "zs":
        args.few_shot_number = 0        
        args.demo_retrieval_method = None
    if args.reason_hint is None:
        args.reason_hint_pos = None
        args.reason_hint_person = None
    if args.tool_aug is None:
        args.tool_desc = None

    assert_gpt35_turbo_16k(args, chat_paradigm="qa_dialogue")
     
    args = get_paths(args)

    print("---------- Generate prompts ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    


