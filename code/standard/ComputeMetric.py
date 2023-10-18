import json
import logging, logging.config
import os
import argparse

import sys
sys.path.append("code")
from utils import get_logger, load_data, save_data, compute_metrics
from utils_parse_answer import two_stage_majority_voting
from const import model_list

logger = logging.getLogger()

def main(args):
    # load response data
    data_response = load_data(args.response_path)

    # compute evaluation results
    compute_metrics(args, data_response)
    
    # saving result file
    # convert responses into string for readability, or each item could be lengthy
    for i in range(len(data_response)):
        data_response[i]["prediction"] = str(data_response[i]["prediction"])
        if args.consistency:
            data_response[i]["responses"] = str(data_response[i]["responses"])
            data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])

    save_data(args.pred_path, data_response)
    logger.info(f"Prediction data saved to: {args.pred_path}")

def get_paths(args):
    dataname = args.dataname

    # label set
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # response loading path
    folder_0 = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed"]:
        folder_0 = f"fs_{folder_0}"    
    if args.few_shot_setting in ["fixed"]:
        folder_0 = f"{folder_0}_{args.demo_select_method}_{args.demo_size}"
    if args.tool_aug:
        folder_0 = f"{folder_0}_tool"
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"

        # SC voting method
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]
    else:
        folder = folder_0

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    datamode = args.datamode

    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ComputeMetric.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    model_folder = model_list[args.model]["abbr"]
    parse_tool_folder = args.parse_tool

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
    # data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
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
    parser.add_argument("--tool_desc", default=1)

    # parsing tool
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])

    # experiment
    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()
    
    assert args.start_time is not None
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

    args = get_paths(args)

    logger.info("\n\n\n---------- Compute Evaluation Results ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)