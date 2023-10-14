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

from utils import get_logger, load_data, save_data, dict2json, format_json2str, assert_gpt35_turbo_16k, run_llm
from utils_parse_answer import response_2_prediction, majority_vote_ner, combine_question_predictions, compute_consistency_score, combine_consistency_scores, collect_mention2labels

from const import dataset_language_map, dataset_label_order_map, my_api_keys

logger = logging.getLogger()

def get_prompt_for_rationale_zh(args, query, label):
    if args.prompt_method == "basic":
        info_labelset = "给定实体标签集：%s\n" % args.id2label
        task_desc = "基于给定的实体标签集，我们需要从给定文本中识别出命名实体。并且我们要以如下JSON格式提供答案：[{\"实体名称\": \"实体标签\"}]。如果文本中没有对应实体，返回如下空列表：[]。 \n"
        info_text = "文本：%s\n" % query["sentence"]
        info_label="答案：%s\n" % label
        # question = "理由如下："
        question = "请给出上述答案的原因。"

        prompt = info_labelset + task_desc + info_text + info_label + question

        return prompt
    else:
        raise ValueError(f"Unrecognized prompt_method: {args.prompt_method}")

def get_prompt_for_rationale_en(args, query, label):
    if args.prompt_method == "basic":
        info_labelset = "Given entity label set: %s\n" % args.id2label
        task_desc = "Based on the given entity label set, we need to recognize the named entities in the given text. And we need provide the answer in the following JSON format: [{\"Entity Name\": \"Entity Label\"}]. If there is no corresponding entity, we return the following empty list: []. \n"
        info_text = "Given text: %s\n" % query["sentence"]
        info_label="The answer is: %s\n" % label
        # question = "The reason is as follows:"
        question = "Please provide the rationale for the above answer."

        prompt = info_labelset + task_desc + info_text + info_label + question

        return prompt
    else:
        raise ValueError(f"Unrecognized prompt_method: {args.prompt_method}")

def get_prompt_for_rationale(args, query, label):
    if args.lang == "en":
        prompt = get_prompt_for_rationale_en(args, query, label)
    elif args.lang == "zh":
        prompt = get_prompt_for_rationale_zh(args, query, label)
    else:
        raise ValueError(f"Unrecognized language: {args.lang}")

    return prompt

def get_rationale(args, pred_data):
    bar = tqdm(pred_data, ncols=100, desc="Generate rationale")
    start_idx = 0

    pred_data_rationale = []
    
    if args.start_time and args.breakpoint_continue:
        pred_data_rationale = load_data(args.demo_gen_ra_path)
        if len(pred_data_rationale) > 0:
            start_idx = len(pred_data_rationale)
        logger.info(f"Continue from last run, start_idx={start_idx}.")
    
    with open(args.demo_gen_ra_path, "ab", buffering=0) as realtime_f:
        for i_query, query in enumerate(bar):

            if i_query < start_idx:
                continue

            # 为label生成rationale
            label = query["label"]
            if isinstance(label, str):
                label = eval(label)

            label_prompt = get_prompt_for_rationale(args, query, label)
            label_messages = [
                {"role": "user", "content": label_prompt}
            ]
            # label_response = run_llm(
            #     label_messages,
            #     openai_key=args.api_key,
            #     model_name=args.model,
            #     temperature=args.temperature,
            # )
            label_response = run_llm(
                label_messages,
                openai_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                stop=args.stop
            )

            query["label_rationale"] = label_response

            realtime_f.write((str(query) +"\n").encode("utf-8"))
            pred_data_rationale.append(query)

            # 打印生成全过程
            if i_query % args.step_show_result == 0:
                logger.info(f"\n\n----- {i_query}-th query -----")
                logger.info("idx: {}".format(query["idx"]))
                logger.info("sentence: {}".format(query["sentence"]))
                logger.info("gold label: {}".format(label))
                logger.info("prompt gold label:\n{}".format(label_prompt))
                logger.info("rationale for label\n{}".format(label_response))

    
    logger.info("Finished!")
    logger.info(f"Rationale saved to: {args.demo_gen_ra_path}")
    logger.info(f"used api_key: {args.api_key}")

    return pred_data_rationale

def main(args):
    # 加载数据
    # data_confident_pred = load_data(args.confident_pred_path)

    # data_confident_pred_rationale = get_rationale(args, data_confident_pred)
    # print(f"len(data_confident_pred_rationale) = {len(data_confident_pred_rationale)}")

    # save_data(args.confident_pred_rationale_path_json, data_confident_pred_rationale)

    demo_data = load_data(args.demo_data_path)
    demo_data_rationale = get_rationale(args, demo_data)
    save_data(args.demo_ra_data_path, demo_data_rationale)
    logger.info(f"demo_ra saved to: {args.demo_ra_data_path}")
        
    demo_parse_data = load_data(args.demo_parse_data_path)
    for item_parse, item in zip(demo_parse_data, demo_data_rationale):
        item_parse["label_rationale"] = item["label_rationale"]
    save_data(args.demo_parse_ra_data_path, demo_parse_data)

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

        demo_ra_filename = "train_ra.json"
        args.demo_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_filename}"
        args.demo_ra_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_ra_filename}"

        demo_parse_filename = "train_parse_hanlp.json"
        demo_parse_ra_filename = "train_parse_hanlp_ra.json"
        args.demo_parse_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_parse_filename}"
        args.demo_parse_ra_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_parse_ra_filename}"

        demo_gen_ra_filename = "train_gen_ra.txt"
        args.demo_gen_ra_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_gen_ra_filename}"

    elif args.few_shot_setting in ["fixed", "pool"]:
        demo_folder = f"demo_{args.few_shot_setting}"

        demo_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}.json"
        demo_ra_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_ra.json"
        args.demo_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_filename}"
        args.demo_ra_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_ra_filename}"

        demo_parse_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"
        demo_parse_ra_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}_ra.json"
        args.demo_parse_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_parse_filename}"
        args.demo_parse_ra_data_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_parse_ra_filename}"

        demo_gen_ra_filename = f"train_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_gen_ra.txt"
        args.demo_gen_ra_path = f"OPENAI/data/{args.task}/{dataname}/{demo_folder}/{demo_gen_ra_filename}"
    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")

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
    
    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    # prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"

    # prompt_dir = f"OPENAI/prompts/{args.task}/{dataname}/{prompt_folder}"
    # if args.parse_tool != "hanlp":
    #     prompt_dir = f"OPENAI/prompts/{args.task}/{args.parse_tool}/{dataname}/{prompt_folder}"
    # if not os.path.exists(prompt_dir):
    #     os.makedirs(prompt_dir)
    # args.save_prompt_path = os.path.join(prompt_dir, prompt_filename)

    # Logger setting
    folder_log = folder  
    log_dir = f"OPENAI/log/{args.task}/{dataname}/{folder_log}"
    logger_filename = f"{args.start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_GPTRationale.log"
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
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--datamode", default="test", type=str)
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=300, type=int)
    # choices=["random_42", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos", "SBERTEmbCos"])
    parser.add_argument("--few_shot_number", default=5, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    
    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)

    # 解析工具选择
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    parser.add_argument("--prompt_method", default=None, type=str)
    parser.add_argument("--step_show_result", default=None, type=int)
    
    # 实验断点续接
    parser.add_argument("--start_time", default=None)
    parser.add_argument("--breakpoint_continue", default=False, action="store_true")

    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]
    # 设置stop列表
    # stop_ls = ["\n", "[]", "[{}]"]
    stop_ls = None
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
    # demo数量超过阈值则切换为16k模型
    assert_gpt35_turbo_16k(args, chat_paradigm="standard")
    # if args.few_shot_setting != "zs":
    #     if args.reason_hint is None and args.tool_aug is None:
    #         if args.few_shot_number >= 20:
    #             assert args.model == "gpt-3.5-turbo-16k"
    #     else:
    #         if args.few_shot_number >= 10:
    #             assert args.model == "gpt-3.5-turbo-16k"

    args = get_paths(args)

    args.api_key = my_api_keys

    if not ("gpt" in args.model) and not ("text" in args.model):
        args.api_key = "EMPTY"

    logger.info("---------- Ask ChatGPT Gen Rationale ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)
