import json
import logging
import os
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

import sys
sys.path.append("code")
from utils import get_logger, load_data, save_data, assert_gpt35_turbo_16k, compute_metrics
from utils_parse_answer import two_stage_majority_voting
from const import dataset_language_map, dataset_label_order_map, model_list

logger = logging.getLogger()


def main(args):
    # load data
    data_response = load_data(args.response_path)

    # compute evaluation metrics
    compute_metrics(args, data_response)

    # saving result file
    # convert responses into string for readability, or each item could be lengthy
    for i in range(len(data_response)):
        data_response[i]["prediction"] = str(data_response[i]["prediction"])
        data_response[i]["responses"] = str(data_response[i]["responses"])
        if args.consistency:
            if args.consis_level == "question":
                data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])
            elif args.consis_level == "sample":
                data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])
                data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
        else:
            data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])

    save_data(args.pred_path, data_response)
    logger.info(f"Prediction data saved to: {args.pred_path}")    


def get_paths(args):
    dataname = args.dataname
    if args.dataname == "ace04en":
        dataname = f"{args.dataname}/{args.folder}"

    # label
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # response loading path
    folder_0 = f"{args.few_shot_setting}"
    if args.tool_aug:
        folder_0 = f"{folder_0}_tool"
    prompt_folder = folder_0
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_{flag_majority_voting}"

        # SC voting
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]
    else:
        folder=folder_0

    # add qa-tag to path
    if args.demo_form == "dialogue":
        qa_tag = "classDia"  
        if args.output_constraint_pos == "behindT":
            qa_tag += "_BT"
    else:
        raise ValueError(f"Unrecognized demo_form = {args.demo_form}")        
    folder = f"{qa_tag}_{args.order}_{folder}"

    prompt_folder = f"{qa_tag}_{args.order}_{prompt_folder}"

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    datamode = args.datamode

    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ComputeMetric.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    model_folder = model_list[args.model]["abbr"]
    parse_tool_folder = args.parse_tool

    args.prompt_path = f"prompts/{parse_tool_folder}/{dataname}/{prompt_folder}/{prompt_filename}"
    args.response_path = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{response_filename}"
    args.response_dir = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder}"
    args.pred_path = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{pred_filename}"
    args.metric_path = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{metric_filename}"
    args.twostage_path = f"result/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{twostage_filename}"

    # Logger setting
    log_dir = f"log/{model_folder}/{parse_tool_folder}/{dataname}/{folder}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
    # Model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    # prompt
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=[None, "first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["zs"])
    parser.add_argument("--demo_size", default=0, type=int)
    parser.add_argument("--few_shot_number", default=0, type=int)
    parser.add_argument("--demo_form", default="dialogue", type=str, choices=["dialogue"])    
    # where to put output format constraint
    parser.add_argument("--output_constraint_pos", default="behindQ", type=str, choices=["behindT", "behindQ"])

    # QA setting    
    parser.add_argument("--order", default=None, type=str)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int, choices=[0,1])
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    # SC voting method
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1)

    # parsing tool
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])

    # experiment
    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]
    prompt_pool_choices = {
        "en": PromptPoolEnglish,
        "zh": PromptPoolChinese
    }
    args.prompt_pool = prompt_pool_choices[args.lang]
    args.label_order = dataset_label_order_map[args.dataname][args.order]

    stop_ls = None
    args.stop = stop_ls
    
    assert args.start_time is not None
    if args.few_shot_setting == "zs":
        args.few_shot_number = 0        
        args.demo_retrieval_method = None
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

    assert_gpt35_turbo_16k(args, chat_paradigm="qa_dialogue")     

    args = get_paths(args)

    if not ("gpt" in args.model) and not ("text" in args.model):
        args.api_key = "EMPTY"

    logger.info("\n\n\n---------- Compute Evaluation Results ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)