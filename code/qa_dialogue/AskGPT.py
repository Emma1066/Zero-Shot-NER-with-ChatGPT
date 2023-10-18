import json
import time
import logging
import sys
import os
from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

import sys
sys.path.append("code")
from utils import get_logger, load_data, assert_gpt35_turbo_16k, run_llm, set_api_key
from utils_parse_answer import response_2_prediction, two_stage_majority_voting, combine_question_predictions, compute_consistency_score, combine_consistency_scores

from const import dataset_language_map, dataset_label_order_map, model_list

logger = logging.getLogger()


def generate_responses_per_query(args, query):
    prompts = query["prompt"]
    task_desc = prompts["task_description"]
    query_info = prompts["query_information"]
    output_constraint = prompts["output_constraint"]
    question_hint = prompts["question_hint"]
    answer_hint = prompts["answer_hint"]
    reason_hint = prompts["reason_hint"]
    demos_prompts = prompts["demos_prompts"]
    questions  = prompts["questions"]
    messages = []
    responses = []
    prediction_per_quest = []
    
    for i_q, quest in enumerate(questions):
        # fetch current question
        if i_q == 0:
            curr_question = task_desc
            # syntactic prompting (front)
            if args.reason_hint_pos=="f":
                curr_question += reason_hint
            ## output format constraint (behind task desc)
            if args.output_constraint_pos == "behindT":
                curr_question += output_constraint
            # demonstration, if having any
            if len(demos_prompts) > 0:
                curr_question += "\n\n" + demos_prompts
            # current question
            curr_question += "\n\n" + query_info
            curr_question += "\n" + question_hint + quest
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            # answer hint
            curr_question += "\n" + answer_hint
            # syntactic prompting (back)
            if args.reason_hint_pos=="b":
                curr_question += reason_hint
        else:
            curr_question = question_hint + quest
            # output format constraint (behind question)
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            if args.output_constraint_pos == "behindT" and args.OC1 == 1:
                curr_question += output_constraint
            # answer hint
            curr_question += "\n" + answer_hint
            # syntactic prompting (back)
            if args.reason_hint_pos=="b":
                curr_question += reason_hint

        messages.append(
            {"role": "user", "content": curr_question}
        )
        # fetch response from LLMs
        response = run_llm(
            messages,
            openai_key=args.api_key,
            model_name=args.model,
            temperature=args.temperature,
            stop=args.stop
        )
        responses.append(response)

        # TODO: debug
        # print(F"\n=== Q ===: {curr_question}")
        # print(F"=== A ===: {response}\n")

        # parse prediction from the responded text
        prediction = response_2_prediction(args, query, response, question=quest, return_form=args.output_format)
        prediction_per_quest.append(prediction)

        messages.append(
            {"role": "assistant", "content": "%s" % prediction}
        )

    # TODO: debug
    # print(f"\nMessages:\n")
    # for tmp_item in messages:
    #     print(tmp_item)
        
    prediction_aggregated = combine_question_predictions(args, prediction_per_quest, return_form=args.output_format)

    query_resp = {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "label": query["label"]
    }
    if args.few_shot_setting == "zs":
        query_resp["prompt"] = query["prompt"]

    query_resp["responses"] = responses
    query_resp["prediction_per_quest"] = prediction_per_quest
    query_resp["prediction"] = prediction_aggregated # dict

    return query_resp

def generate_responses_per_query_consist_question(args, query, query_times=5, temperature=0.7):
    prompts = query["prompt"]
    task_desc = prompts["task_description"]
    query_info = prompts["query_information"]
    output_constraint = prompts["output_constraint"]
    question_hint = prompts["question_hint"]
    answer_hint = prompts["answer_hint"]
    reason_hint = prompts["reason_hint"]
    demos_prompts = prompts["demos_prompts"]
    questions  = prompts["questions"]
    messages = []
    responses = []
    prediction_per_quest = []
    consistency_scores = []
    for i_q, quest in enumerate(questions):
        if i_q == 0:
            curr_question = task_desc
            # syntactic prompting (front)
            if args.reason_hint_pos=="f":
                curr_question += reason_hint
            # output format constraint (behind task desc)
            if args.output_constraint_pos == "behindT":
                curr_question += output_constraint
            # demonstrations, if having any
            if len(demos_prompts) > 0:
                curr_question += "\n\n" + demos_prompts
            # current question
            curr_question += "\n\n" + query_info
            curr_question += "\n" + question_hint + quest
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            # answer hint
            curr_question += "\n" + answer_hint
            # syntactic prompting (back)
            if args.reason_hint_pos=="b":
                curr_question += reason_hint
        else:
            curr_question = question_hint + quest
            # output format constraint (behind question)
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            # answer hint
            curr_question += "\n" + answer_hint
            # syntactic prompting (back)
            if args.reason_hint_pos=="b":
                curr_question += reason_hint

        messages.append(
            {"role": "user", "content": curr_question}
        )
        # sample multiple responses
        curr_responses = []
        curr_predictions = []
        for i_consist in range(query_times):
            tmp_response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            
            curr_responses.append(tmp_response)
            # current prediction
            tmp_prediction = response_2_prediction(args, query, tmp_response, question=quest, return_form=args.output_format)
            curr_predictions.append(tmp_prediction)

            # TODO: debug
            # print(F"\n=== Q ===:\n{curr_question}")
            # print(F"=== A ===:\n{tmp_response}\n")
        
        responses.append(curr_responses)
        # voted prediction for current question
        MV_func = args.MV_func
        curr_prediction_voted = MV_func(args, curr_predictions)
        prediction_per_quest.append(curr_prediction_voted)
        # SC score of the voted prediction for current question
        curr_consistency_score = compute_consistency_score(curr_predictions, curr_prediction_voted)
        consistency_scores.append(curr_consistency_score)
        # context for next round
        messages.append(
            {"role": "assistant", "content": "%s" % curr_prediction_voted}
        )

    # prediction for all questions
    prediction_aggregated = combine_question_predictions(args, prediction_per_quest, return_form=args.output_format)

    # SC score for final answers
    consistency_score_entities = combine_consistency_scores(consistency_scores, prediction_aggregated)
    if len(consistency_score_entities) == 0:
        consistency_score_avg = 0
    else:
        consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)

    query_resp = {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "label": query["label"]
    }
    if args.few_shot_setting == "zs":
        query_resp["prompt"] = query["prompt"]

    query_resp["responses"] = responses # all response
    query_resp["prediction_per_quest"] = prediction_per_quest
    query_resp["prediction"] = prediction_aggregated # final answer
    query_resp["consistency_score"] = {"entities":consistency_score_entities, "avg":consistency_score_avg} # SC score all final answer

    return query_resp

def generate_responses_per_query_consist_sample(args, query, query_times=5, temperature=0.7):
    prompts = query["prompt"]
    task_desc = prompts["task_description"]
    query_info = prompts["query_information"]
    output_constraint = prompts["output_constraint"]
    question_hint = prompts["question_hint"]
    answer_hint = prompts["answer_hint"]
    reason_hint = prompts["reason_hint"]
    demos_prompts = prompts["demos_prompts"]
    questions  = prompts["questions"]
    responses = []
    prediction_per_quest = []
    prediction_aggregated_per_consist = []
    for i_consist in range(query_times):
        messages = []
        curr_responses = []
        curr_prediction_per_quest = []
        for i_q, quest in enumerate(questions):
            if i_q == 0:
                curr_question = task_desc
                # syntactic prompting (front)
                if args.reason_hint_pos=="f":
                    curr_question += reason_hint
                # output format constraint (behind task desc)
                if args.output_constraint_pos == "behindT":
                    curr_question += output_constraint
                # demonstrations, if having any
                if len(demos_prompts) > 0:
                    curr_question += "\n\n" + demos_prompts
                # current question
                curr_question += "\n\n" + query_info
                curr_question += "\n" + question_hint + quest
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                # answer hint
                curr_question += "\n" + answer_hint
                # # syntactic prompting (back)
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint
            else:
                curr_question = question_hint + quest
                # output format constraint (behind question)
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                # answer hint
                curr_question += "\n" + answer_hint
                # # syntactic prompting (back)
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint
            
            messages.append(
                {"role": "user", "content": curr_question}
            )
            # fetch response from LLM
            tmp_response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            curr_responses.append(tmp_response)
    
            # parse prediction from the responded text
            tmp_prediction = response_2_prediction(args, query, tmp_response, question=quest, return_form=args.output_format)
            curr_prediction_per_quest.append(tmp_prediction)

            messages.append(
                {"role": "assistant", "content": "%s" % tmp_prediction}
            )

        prediction_per_quest.append(curr_prediction_per_quest)
        
        curr_prediction_aggregated = combine_question_predictions(args, curr_prediction_per_quest, return_form=args.output_format)
        
        responses.append(curr_responses)
        prediction_aggregated_per_consist.append(curr_prediction_aggregated)
    
        # TODO: debug
        # print(F"\n=== Q ===:\n{curr_question}")
        # print(F"=== A ===:\n{tmp_response}\n")

    # voted on all predictions
    MV_func = args.MV_func
    prediction_aggregated = MV_func(args, prediction_aggregated_per_consist)

    # SC scores for the voted answer
    consistency_score_entities = compute_consistency_score(prediction_aggregated_per_consist, prediction_aggregated)
    if len(consistency_score_entities) > 0:
        consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
    else:
        consistency_score_avg = 0

    query_resp = {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "label": query["label"]
    }
    if args.few_shot_setting == "zs":
        query_resp["prompt"] = query["prompt"]

    query_resp["responses"] = responses # all responss: all repeated dialogue, all questions
    query_resp["prediction_per_quest"] = prediction_per_quest # all responss: all repeated dialogue, all questions
    query_resp["prediction_per_consist"] = prediction_aggregated_per_consist # all responss: all repeated dialogue, all questions
    query_resp["prediction"] = prediction_aggregated # final voted prediction
    query_resp["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg} # SC scores for the voted answer
    
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

            # if not started from the first sample (e.g, interruption due to network connection)
            if i_query < start_idx:
                continue
            if not args.consistency:
                query_resp = generate_responses_per_query(args, query)
            else:
                if args.consis_level == "question":
                    query_resp = generate_responses_per_query_consist_question(args, query, query_times=args.query_times, temperature=args.temperature)
                elif args.consis_level == "sample":
                    query_resp = generate_responses_per_query_consist_sample(args, query, query_times=args.query_times, temperature=args.temperature)
                else:
                    raise ValueError(f"Unrecognized consist_level: {args.consis_level}")
            
            realtime_f.write((str(query_resp)+"\n").encode("utf-8"))
    
    logger.info("Finished!")
    logger.info(f"response saved to: {args.response_path}")
    logger.info(f"used api_key: {args.api_key}")
        

def main(args):
    # load data (prompt)
    data_prompts = load_data(args.prompt_path)
    
    # asking LLM
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
    if args.tool_aug:
        folder = f"{folder}_tool"
    # add qa-tag into path
    if args.demo_form == "dialogue":
        qa_tag = "classDia"
        if args.output_constraint_pos == "behindT":
            qa_tag += "_BT"
    else:
        raise ValueError(f"Unrecognized demo_form = {args.demo_form}")           
    folder = f"{qa_tag}_{args.order}_{folder}"

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
        folder_resp = f"{folder_resp}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_{flag_majority_voting}"
        # SC voting function
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
    # Data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
    # Model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--ports", default=None, nargs="+", type=int)

    # prompt
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=[None, "first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["zs"])
    parser.add_argument("--demo_size", default=0, type=int)
    parser.add_argument("--few_shot_number", default=0, type=int)
 
    # QA setting
    parser.add_argument("--order", default=None, type=str)
    parser.add_argument("--demo_form", default="dialogue", type=str, choices=["dialogue"])

    # output format
    parser.add_argument("--output_format", default="json", type=str, choices=["json", "list"])
    # where to put output format constraint
    parser.add_argument("--output_constraint_pos", default="behindQ", type=str, choices=["behindT", "behindQ"])

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int, choices=[0,1])
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    # SC voting method
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)
    
    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # parsing tools
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp"])

    # maybe continue from a interruption
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--breakpoint_continue", default=False, action="store_true")

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

    args.api_key = set_api_key(model_name=args.model, ports=args.ports)
    
    args = get_paths(args)

    logger.info("---------- Ask ChatGPT ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)