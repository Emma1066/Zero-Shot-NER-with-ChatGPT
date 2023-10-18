import json
import time
import logging
import os

from tqdm import tqdm
import argparse

import sys
sys.path.append("code")
from utils import get_logger, load_data, assert_gpt35_turbo_16k, run_llm, set_api_key
from utils_parse_answer import response_2_prediction, two_stage_majority_voting, compute_consistency_score
from const import model_list

logger = logging.getLogger()

def generate_responses_per_query(args, query):
    messages = [
        {"role": "user", "content": query["prompt"]}
    ]
    response = run_llm(
        messages,
        openai_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature,
        stop=args.stop
    )

    query_resp = {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "label": query["label"]
    }
    if args.few_shot_setting == "zs":
        query_resp["prompt"] = query["prompt"]

    query_resp["response"] = response
    query_resp["prediction"] = response_2_prediction(args, query, response)

    return query_resp

def generate_responses_per_query_multiquery(args, query, query_times=5, temperature=1.0):
    messages = [
        {"role": "user", "content": query["prompt"]}
    ]

    query_resp = {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "label": query["label"]
    }    
    if args.few_shot_setting == "zs":
        query_resp["prompt"] = query["prompt"]    

    responses = []
    predictions = []
    for i_time in range(query_times):
        response = run_llm(
            messages,
            openai_key=args.api_key,
            model_name=args.model,
            temperature=temperature,
            stop=args.stop
        )
        responses.append(response)
        predictions.append(response_2_prediction(args, query, response))

    query_resp["responses"] = responses
    query_resp["prediction_per_consist"] = predictions
    
    # SC voting method
    MV_func = args.MV_func
    prediction_voted = MV_func(args, predictions)
    query_resp["prediction"] = prediction_voted

    consistency_score_entities = compute_consistency_score(predictions, prediction_voted)
    if len(consistency_score_entities):
        consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
    else:
        consistency_score_avg = 0
    query_resp["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg}

    return query_resp

def generate_responses_batch(args, data_prompts):
    bar = tqdm(data_prompts, ncols=100)
    start_idx = 0
    if args.start_time and args.breakpoint_continue:
        pre_res = load_data(args.response_path)
        if len(pre_res) > 0:
            start_idx = len(pre_res)
    with open(args.response_path, "ab", buffering=0) as realtime_f:
        for i_query, query in enumerate(bar):
            bar.set_description("Query ChatGPT NER")

            # if not started from the first sample
            if i_query < start_idx:
                continue
            
            if not args.consistency:
                query_resp = generate_responses_per_query(args, query)
            else:
                query_resp = generate_responses_per_query_multiquery(args, query, query_times=args.query_times, temperature=args.temperature)
            
            realtime_f.write((str(query_resp)+"\n").encode("utf-8"))
    
    logger.info("Finished!")
    logger.info(f"response saved to: {args.response_path}")
    logger.info(f"used api_key: {args.api_key}")
        

def main(args):
    # load data
    data_prompts = load_data(args.prompt_path)
    
    # obtain LLM response
    generate_responses_batch(
        args, 
        data_prompts
    )

def get_paths(args):
    dataname = args.dataname

    # label set
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())
    
    # prompt loading path
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed"]:
        folder = f"fs_{folder}"
    if args.few_shot_setting in ["fixed"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
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
    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_AskGPT.log"

    model_folder = model_list[args.model]["abbr"]
    parse_tool_folder = args.parse_tool

    args.prompt_path = f"prompts/{parse_tool_folder}/{dataname}/{folder}/{prompt_filename}"

    folder_resp = folder
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder_resp = f"{folder_resp}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"
        
        # SC voting method
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]

    response_dir = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder_resp}"
    if not os.path.exists(response_dir):
        os.makedirs(response_dir)    
    args.response_path = os.path.join(response_dir, response_filename)

    # Logger setting
    folder_log = folder_resp
    log_dir = f"log/{model_folder}/{parse_tool_folder}/{dataname}/{folder_log}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--ports", default=None, nargs="+", type=int)

    # prompt
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "zs"])
    parser.add_argument("--demo_size", default=3, type=int)
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--few_shot_number", default=3, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int, choices=[0,1])
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)
    
    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # parsing tool
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])

    # experiment
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--breakpoint_continue", default=False, action="store_true")

    args = parser.parse_args()

    stop_ls = None
    args.stop = stop_ls
    
    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
    if args.few_shot_setting == "zs":
        args.few_shot_number = 0 

    if args.reason_hint is None:
        args.reason_hint_pos = None
        args.reason_hint_person = None

    if args.tool_aug is None:
        args.tool_desc = None        

    if not args.consistency:
        args.temperature = 0

    if args.consistency:
        assert args.temperature > 0
        assert args.query_times > 0
    else:
        assert args.temperature == 0

    assert_gpt35_turbo_16k(args, chat_paradigm="standard")

    args.api_key = set_api_key(model_name=args.model, ports=args.ports)

    args = get_paths(args)

    logger.info("---------- Ask ChatGPT ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)