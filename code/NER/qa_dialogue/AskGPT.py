import json
import time
import logging, logging.config
import sys
import os
from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

from os import path
import sys
# 导入parent目录下的模块
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import get_logger, load_data, dict2json, format_json2str, assert_gpt35_turbo_16k, run_llm, set_api_key
from utils_parse_answer import response_2_prediction, two_stage_majority_voting, majority_voting, combine_question_predictions, collect_mention2labels, compute_consistency_score, combine_consistency_scores

from const import dataset_language_map, dataset_label_order_map, model_list

logger = logging.getLogger()


def generate_responses_per_query(args, query):
    prompts = query["prompt"]
    task_desc = prompts["task_description"]
    query_info = prompts["query_information"]
    output_constraint = prompts["output_constraint"]
    # question_hint = prompts["question_hint"]
    # answer_hint = prompts["answer_hint"]
    # reason_hint = prompts["reason_hint"]
    demos_prompts = prompts["demos_prompts"]
    questions  = prompts["questions"]
    messages = []
    responses = []
    prediction_per_quest = []
    # few-shot模式，补全整个dialogue
    # TODO: 将answer_hint, output_constraint_pos, reason_hint都分解
    if args.few_shot_setting!="zs" and args.demo_form=="dialogue" and args.complete_dialogue==1:
        # question = ""
        # question += task_desc
        # if args.output_constraint_pos == "behindT":
        #     question += output_constraint_json
        # question += "\n\n" + demos_prompts
        # question += "\n\n" + query_info

        question = task_desc
        # # 推理
        # if args.reason_hint_pos=="f":
        #     question += reason_hint
        # 输出格式限制curr_question
        # 暂时只考虑behindT情况
        assert args.output_constraint_pos == "behindT"
        if args.output_constraint_pos == "behindT":
            question += output_constraint
        # 样例
        if len(demos_prompts) > 0:
            question += "\n\n" + demos_prompts
        # 当前要测试的query问题
        question += "\n\n" + query_info
        # # 推理
        # if args.reason_hint_pos=="b":
        #     question += reason_hint

        messages.append({"role": "user", "content": question})
        response = run_llm(
            messages,
            openai_key=args.api_key,
            model_name=args.model,
            temperature=args.temperature,
            stop=args.stop
        )
        responses.append(response)

        # TODO: debug
        # print(question)
        # print(response)

        prediction_per_quest, responded_questions, responded_answers = response_2_prediction(args, query, response, complete_form="dialogue", return_responded_qa=True)
        
        # 将修正后的ChatGPT补全的对话加入到messages列表里
        if len(responded_questions)==0:
            responded_qas=""
        else:
            responded_qas = responded_questions[0] + "\n" + responded_answers[0]
            for q, a in zip(responded_questions[1:], responded_answers[1:]):
                responded_qas += "\n" + q + "\n" + a
        messages.append({"role": "assistant", "content": responded_qas})
    else:
        for i_q, quest in enumerate(questions):
            # 获取当前问题
            if i_q == 0:
                curr_question = task_desc
                # 推理
                if args.reason_hint_pos=="f":
                    curr_question += reason_hint
                # 输出格式限制curr_question
                if args.output_constraint_pos == "behindT":
                    curr_question += output_constraint
                # 样例
                if len(demos_prompts) > 0:
                    curr_question += "\n\n" + demos_prompts
                # 当前要测试的query问题
                curr_question += "\n\n" + query_info
                curr_question += "\n" + question_hint + quest
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                # 答案提示
                curr_question += "\n" + answer_hint
                # 推理
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint
            else:
                curr_question = question_hint + quest
                # 输出格式限制curr_question
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                if args.output_constraint_pos == "behindT" and args.OC1 == 1:
                    curr_question += output_constraint
                # 答案提示
                curr_question += "\n" + answer_hint
                # 推理
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint

            messages.append(
                {"role": "user", "content": curr_question}
            )
            # 获取当前回答
            response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            responses.append(response)

            # TODO: debug
            # print(F"=== Q ===: {curr_question}")
            # print(F"=== A ===: {response}")
    
            # 获取当前预测结果
            prediction = response_2_prediction(args, query, response, question=quest, return_form=args.output_format)
            prediction_per_quest.append(prediction)

            # TODO: debug,用raw response还是prediction
            messages.append(
                {"role": "assistant", "content": format_json2str(dict2json(prediction))}
            )
        
    # label消歧：一个实体只能属于一个标签
    if args.label_disambiguation == 1:
        prediction_all_json = [] # 用以label消歧
        for tmp_pred in prediction_per_quest:
            prediction_all_json.extend(dict2json(tmp_pred))
        mention2labels = collect_mention2labels(prediction_all_json)
        prediction_aggregated = {} # 所有问题预测结果聚合、消歧的结果
        for tmp_mention, tmp_label_list in mention2labels.items():
            if len(tmp_label_list) == 1:
                prediction_aggregated[tmp_mention] = tmp_label_list[0]
                continue
            prompt_label_disambiguation = args.prompt_pool.prompt_label_disambiguation(tmp_mention, tmp_label_list)
            messages.append({"role": "user", "content": prompt_label_disambiguation})
            # 获取当前实体label消歧的回答
            response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            responses.append(response)
            messages.append({"role": "assistant", "content": response})
            prediction_aggregated[tmp_mention] = response
    else:
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
    # SC所有出现过的答案的频次
    cnt_prediction_tuple = {}
    # TODO: few-shot模式，补全整个dialogue ########
    for i_q, quest in enumerate(questions):
        if i_q == 0:
            curr_question = task_desc
            # 推理
            if args.reason_hint_pos=="f":
                curr_question += reason_hint
            # 输出格式限制curr_question
            if args.output_constraint_pos == "behindT":
                curr_question += output_constraint
            # 样例
            if len(demos_prompts) > 0:
                curr_question += "\n\n" + demos_prompts
            # 当前要测试的query问题
            curr_question += "\n\n" + query_info
            curr_question += "\n" + question_hint + quest
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            # 答案提示
            curr_question += "\n" + answer_hint
            # 推理
            if args.reason_hint_pos=="b":
                curr_question += reason_hint
        else:
            curr_question = question_hint + quest
            # 输出格式限制curr_question
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint
            if args.output_constraint_pos == "behindT" and args.OC1 == 1:
                curr_question += output_constraint
            # 答案提示
            curr_question += "\n" + answer_hint
            # 推理
            if args.reason_hint_pos=="b":
                curr_question += reason_hint

        messages.append(
            {"role": "user", "content": curr_question}
        )
        # 获取当前多个回答
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
            # 获取当前预测结果
            tmp_prediction = response_2_prediction(args, query, tmp_response, question=quest, return_form=args.output_format)
            curr_predictions.append(tmp_prediction)
            # 统计所有预测中出现的tuple频次
            for k,v in tmp_prediction.items():
                if (k,v) not in cnt_prediction_tuple:
                    cnt_prediction_tuple[(k,v)] = 0
                cnt_prediction_tuple[(k,v)] += 1
        
        responses.append(curr_responses)
        # 获取当前问题预测结果的vote结果
        MV_func = args.MV_func
        curr_prediction_voted = MV_func(args, curr_predictions)
        prediction_per_quest.append(curr_prediction_voted)
        # 计算vote结果的consistency score
        curr_consistency_score = compute_consistency_score(curr_predictions, curr_prediction_voted)
        consistency_scores.append(curr_consistency_score)
        # 准备下一轮问答的context
        messages.append(
            {"role": "assistant", "content": format_json2str(dict2json(curr_prediction_voted))}
        )

    # 获取所有问题预测结果的聚合结果
    # label消歧：一个实体只能属于一个标签
    if args.label_disambiguation == 1:
        prediction_all_json = [] # 用以label消歧
        for tmp_pred in prediction_per_quest:
            prediction_all_json.extend(dict2json(tmp_pred))
        mention2labels = collect_mention2labels(prediction_all_json)
        prediction_aggregated = {} # 所有问题预测结果聚合、消歧的结果
        for tmp_mention, tmp_label_list in mention2labels.items():
            if len(tmp_label_list) == 1:
                prediction_aggregated[tmp_mention] = tmp_label_list[0]
                continue
            prompt_label_disambiguation = args.prompt_pool.prompt_label_disambiguation(tmp_mention, tmp_label_list)
            messages.append({"role": "user", "content": prompt_label_disambiguation})
            # 获取当前实体label消歧的回答
            response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            responses.append(response)
            messages.append({"role": "assistant", "content": response})
            prediction_aggregated[tmp_mention] = response
    else:
        prediction_aggregated = combine_question_predictions(args, prediction_per_quest, return_form=args.output_format)

    # 根据最终聚合的prediction获取对应{mention:type}的consistency score
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

    query_resp["responses"] = responses # 所有response：所有question，所有sampled responses
    query_resp["prediction_per_quest"] = prediction_per_quest # 每个问题voted后的结果
    query_resp["prediction"] = prediction_aggregated # 所有问题预测聚合后的最终预测
    query_resp["consistency_score"] = {"entities":consistency_score_entities, "avg":consistency_score_avg} # 最终投票出的答案中每个实体的consistency score (dict)
    # 所有SC答案的score
    if len(cnt_prediction_tuple):
        avg_cnt_prediction_tuple = sum(list(cnt_prediction_tuple.values())) / len(cnt_prediction_tuple)
    else:
        avg_cnt_prediction_tuple = 0
    query_resp["consistency_score_SC_all_ans"] = {"entities":cnt_prediction_tuple, "avg":avg_cnt_prediction_tuple}

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
    # SC所有出现过的答案的频次
    cnt_prediction_tuple = {}
    # TODO: # few-shot模式，可以选择补全整个dialogue
    for i_consist in range(query_times):
        messages = []
        curr_responses = []
        curr_prediction_per_quest = []
        for i_q, quest in enumerate(questions):
            # curr_question = quest
            # if i_q == 0:
            #     curr_question = task_desc
            #     if args.output_constraint_pos == "behindT":
            #         curr_question += output_constraint_json
            #     if len(demos_prompts) > 0:
            #         curr_question += "\n\n" + demos_prompts
            #     curr_question += "\n\n" + query_info
            #     curr_question += "\n" + quest

            if i_q == 0:
                curr_question = task_desc
                # 推理
                if args.reason_hint_pos=="f":
                    curr_question += reason_hint
                # 输出格式限制curr_question
                if args.output_constraint_pos == "behindT":
                    curr_question += output_constraint
                # 样例
                if len(demos_prompts) > 0:
                    curr_question += "\n\n" + demos_prompts
                # 当前要测试的query问题
                curr_question += "\n\n" + query_info
                curr_question += "\n" + question_hint + quest
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                # 答案提示
                curr_question += "\n" + answer_hint
                # 推理
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint
            else:
                curr_question = question_hint + quest
                # 输出格式限制curr_question
                if args.output_constraint_pos == "behindQ":
                    curr_question += output_constraint
                if args.output_constraint_pos == "behindT" and args.OC1 == 1:
                    curr_question += output_constraint
                # 答案提示
                curr_question += "\n" + answer_hint
                # 推理
                if args.reason_hint_pos=="b":
                    curr_question += reason_hint
            
            messages.append(
                {"role": "user", "content": curr_question}
            )
            # 获取当前回答
            tmp_response = run_llm(
                messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )
            curr_responses.append(tmp_response)
    
            # 获取当前预测结果
            tmp_prediction = response_2_prediction(args, query, tmp_response, question=quest, return_form=args.output_format)
            curr_prediction_per_quest.append(tmp_prediction)

            # TODO: debug,用raw response还是prediction
            messages.append(
                {"role": "assistant", "content": format_json2str(dict2json(tmp_prediction))}
            )

        prediction_per_quest.append(curr_prediction_per_quest)
        
        # 获取当前所有问题聚合的最终预测
        # label消歧：一个实体只能属于一个标签
        if args.label_disambiguation == 1:
            curr_prediction_all_json = [] # 用以label消歧
            for tmp_pred in curr_prediction_per_quest:
                curr_prediction_all_json.extend(dict2json(tmp_pred))
            curr_mention2labels = collect_mention2labels(curr_prediction_all_json)
            curr_prediction_aggregated = {} # 所有问题预测结果聚合、消歧的结果
            for tmp_mention, tmp_label_list in curr_mention2labels.items():
                if len(tmp_label_list) == 1:
                    curr_prediction_aggregated[tmp_mention] = tmp_label_list[0]
                    continue
                prompt_label_disambiguation = args.prompt_pool.prompt_label_disambiguation(tmp_mention, tmp_label_list)
                messages.append({"role": "user", "content": prompt_label_disambiguation})
                # 获取当前实体label消歧的回答
                tmp_response = run_llm(
                    messages,
                    openai_key=args.api_key,
                    model_name=args.model,
                    temperature=args.temperature,
                    stop=args.stop
                )
                curr_responses.append(tmp_response)
                messages.append({"role": "assistant", "content": tmp_response})
                curr_prediction_aggregated[tmp_mention] = tmp_response
        else:
            curr_prediction_aggregated = combine_question_predictions(args, curr_prediction_per_quest, return_form=args.output_format)       
        
        responses.append(curr_responses)
        prediction_aggregated_per_consist.append(curr_prediction_aggregated)

        # 统计所有预测中出现的tuple频次
        for k,v in curr_prediction_aggregated.items():
            if (k,v) not in cnt_prediction_tuple:
                cnt_prediction_tuple[(k,v)] = 0
            cnt_prediction_tuple[(k,v)] += 1

    # 获取所有问题预测结果的vote结果
    MV_func = args.MV_func
    prediction_aggregated = MV_func(args, prediction_aggregated_per_consist)

    # 根据最终聚合的prediction获取对应{mention:type}的consistency score
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

    query_resp["responses"] = responses # 所有response：所有sampled path, 所有quest
    query_resp["prediction_per_quest"] = prediction_per_quest # 所有sampled path, 所有quest
    query_resp["prediction_per_consist"] = prediction_aggregated_per_consist # 所有sampled path, 所有quest
    query_resp["prediction"] = prediction_aggregated # 最终prediction
    query_resp["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg} # 最终投票出的答案中每个实体的consistency score (dict)
    # 所有SC答案的score
    if len(cnt_prediction_tuple):
        avg_cnt_prediction_tuple = sum(list(cnt_prediction_tuple.values())) / len(cnt_prediction_tuple)
    else:
        avg_cnt_prediction_tuple = 0
    query_resp["consistency_score_SC_all_ans"] = {"entities":cnt_prediction_tuple, "avg":avg_cnt_prediction_tuple}
    
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

            # 如果不是从第一条数据开始
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
    # 加载数据
    data_prompts = load_data(args.prompt_path)
    
    # 获取ChatGPT回答
    generate_responses_batch(
        args, 
        data_prompts
    )
    
def get_paths(args):
    dataname = args.dataname
    if args.dataname == "ace04en":
        dataname = f"{args.dataname}/{args.folder}"

    # 标签集路径 + 加载
    args.abb2labelname_path = f"OPENAI/data/{args.task}/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())
    
    # prompt加载路径
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder = f"fs_{folder}"
    if args.few_shot_setting in ["fixed", "pool"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder = f"{folder}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder = f"{folder}_tool"
    # 加上按类别提问的标记
    if args.demo_form == "entity":
        qa_tag = "classEnt"
    elif args.demo_form == "dialogue":
        qa_tag = "classDia"
        if args.complete_dialogue == 1:
            qa_tag = "classDiaCDia"
        if args.output_constraint_pos == "behindT":
            qa_tag += "_BT"
            if args.OC1==1:
                qa_tag += "_OC1"
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
    # if args.dataname == "conll2003":
    #     datamode = "conllpp_test"
    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_AskGPT.log"

    model_folder = model_list[args.model]["abbr"]
    parse_tool_folder = args.parse_tool

    args.prompt_path = f"OPENAI/prompts/{args.task}/{parse_tool_folder}/{dataname}/{folder}/{prompt_filename}"

    folder_resp = folder
    if args.label_disambiguation==1:
        folder_resp += "_LD"

    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder_resp = f"{folder_resp}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_{flag_majority_voting}"
        # SC投票方式
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting,
                            "majority_voting": majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]

    response_dir = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder_resp}"

    if not os.path.exists(response_dir):
        os.makedirs(response_dir)
    args.response_path = os.path.join(response_dir, response_filename)

    # Logger setting
    folder_log = folder_resp
    log_dir = f"OPENAI/log/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder_log}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"OPENAI/config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--folder", default=0, type=int) # only for ace04
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--ports", default=None, nargs="+", type=int)

    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=3, type=int)
    parser.add_argument("--demo_select_method", default="GPTEmbClusterKmeans") # , choices=["random", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)
 
    # QA setting
    parser.add_argument("--order", default=None, type=str) # 0,1,chatgpt0, chatgpt1
     # qa模式下，demo的形式：(1) 整段conversation; (2) 每个类别的几个entity例子。
    parser.add_argument("--demo_form", default="dialogue", type=str, choices=["dialogue", "entity"])
    # qa few-shot模式下，两种补全形式：(1)和zero-shot一致的问答；(2)补全整个对话
    parser.add_argument("--complete_dialogue", default=0, type=int, choices=[0, 1])
    # output format
    parser.add_argument("--output_format", default="json", type=str, choices=["json", "list"])
    # output constraint的位置
    parser.add_argument("--output_constraint_pos", default="behindQ", type=str, choices=["behindT", "behindQ"])
    parser.add_argument("--OC1", default=0, type=int, choices=[0,1])
    # 是否进行label消歧
    parser.add_argument("--label_disambiguation", default=0, type=int, choices=[0, 1])

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    # SC筛选答案的方法: [two_stage_majority_voting, ]
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)
    
    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # 解析工具选择
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    # 实验断点续接
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--breakpoint_continue", default=False, action="store_true")

    args = parser.parse_args()
    
    # 设置prompt_pool，用以在AskGPT过程中动态生成所需prompt
    args.lang = dataset_language_map[args.dataname]
    prompt_pool_choices = {
        "en": PromptPoolEnglish,
        "zh": PromptPoolChinese
    }
    args.prompt_pool = prompt_pool_choices[args.lang]
    args.label_order = dataset_label_order_map[args.dataname][args.order]
    
    # 设置stop列表
    # stop_ls = ["\n", "[]", "[{}]"]
    # stop_ls = ["[]", "[{}]"]
    stop_ls = ["Question:"] # 防止text-davinci提前生成好下一个问题
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

    # 根据最大上下文长度需求，更改模型
    assert_gpt35_turbo_16k(args, chat_paradigm="qa_dialogue")
    # 设置api keys
    args.api_key = set_api_key(model_name=args.model, ports=args.ports)
    
    args = get_paths(args)

    logger.info("---------- Ask ChatGPT ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)