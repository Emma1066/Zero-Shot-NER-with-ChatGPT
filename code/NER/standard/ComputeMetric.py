import json
import time
import logging, logging.config
import os
import re
import pandas as pd
from collections import Counter

from tqdm import tqdm
import argparse

from os import path
import sys
# 导入parent目录下的模块
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import get_logger, load_data, save_data, json2dict, copy_file_to_path
from utils_parse_answer import response_2_prediction, two_stage_majority_voting, majority_voting, collect_all_SC_answers, compute_consistency_score, parse_response_std
from const import model_abb_map

logger = logging.getLogger()

def compute_metrics(args, data_responses):
    class_n_r = pd.read_csv(args.class_n_r_path)
    mode = args.datamode
    se_type = list(class_n_r["Type"])
    se_count = list(class_n_r[f"{mode}_count"])
    se_rate = list(class_n_r[f"{mode}_rate"])
    se_lmin = list(class_n_r[f"{mode}_len_min"])
    se_lmax = list(class_n_r[f"{mode}_len_max"])
    se_lmean = list(class_n_r[f"{mode}_len_mean"])
    type2count = dict(zip(se_type, se_count))
    type2rate = dict(zip(se_type, se_rate))
    type2lmin = dict(zip(se_type, se_lmin))
    type2lmax = dict(zip(se_type, se_lmax))
    type2lmean = dict(zip(se_type, se_lmean))
    tot_lmin = min(type2lmin.values())
    tot_lmax = min(type2lmax.values())
    tot_lmean = sum([list(type2lmean.values())[i] * list(type2rate.values())[i] for i in range(len(type2rate))])
    tot_lmean = round(tot_lmean, 1)    

    id2label = args.id2label
    n_correct = 0
    n_gold = 0
    n_pred = 0
    n_correct_classwise = dict(zip(id2label, [0]*len(id2label)))
    n_gold_classwise = dict(zip(id2label, [0]*len(id2label)))
    n_pred_classwise = dict(zip(id2label, [0]*len(id2label)))
    ood_type_set = []

    n_correct_span = 0
    for item in tqdm(data_responses, desc="compute mmetric"):
        curr_label = item["label"]
        if isinstance(curr_label, str):
            curr_label = eval(curr_label)
        # 将标注的json格式转换为dict格式
        if isinstance(curr_label, list):
            curr_label = json2dict(curr_label)
        curr_pred = item["prediction"]
        if isinstance(curr_pred, str):
            curr_pred = eval(curr_pred)        

        # 去掉 空字符“” 的预测
        if "" in curr_pred:
            del curr_pred[""]

        # Traveling gold and pred
        for tmp_span_gold, tmp_type_gold in curr_label.items():
            n_gold += 1
            n_gold_classwise[tmp_type_gold] += 1
        for tmp_span_pred, tmp_type_pred in curr_pred.items():
            # 如果span不来自于文本，则不计入统计
            if tmp_span_pred not in item["sentence"]:
                continue            
            n_pred += 1
            if tmp_type_pred not in n_pred_classwise:
                n_pred_classwise[tmp_type_pred] = 0
            n_pred_classwise[tmp_type_pred] += 1
            if tmp_span_pred in curr_label:
                # 仅判断loc是否正确
                n_correct_span += 1
                # 判断loc和typing是否都正确
                tmp_type_gold = curr_label[tmp_span_pred]
                if tmp_type_gold == tmp_type_pred:
                    n_correct += 1
                    n_correct_classwise[tmp_type_gold] += 1      

    # print ood type
    for k in n_pred_classwise:
        if k not in id2label:
            print("{}: {}".format(k, n_pred_classwise[k]))      

    metrics_table = pd.DataFrame(columns=['Label', 'Count', 'Rate', 'Len_min', 'Len_max', 'Len_mean', 'Prec.', 'Rec.', 'F1'])
    # --- compute overall metrics ---
    # both span and type correct
    prec = round(n_correct / n_pred * 100, 2) if n_pred else 0
    rec = round(n_correct / n_gold *100, 2) if n_gold else 0
    f1 = round(2 * prec * rec / (prec + rec), 2) if n_correct else 0
    # span correct, regardless of type
    span_prec = round(n_correct_span / n_pred * 100, 2) if n_pred else 0
    span_rec = round(n_correct_span / n_gold * 100, 2) if n_gold else 0
    span_f1 = round(2 * span_prec * span_rec / (span_prec + span_rec), 2) if n_correct_span else 0
    # based on correct span, type acc
    type_acc = round(n_correct / n_correct_span * 100, 2) if n_correct_span else 0            

    # --- compute classwise metrics ---
    prec_classwise = dict(zip(id2label, [0]*len(id2label)))
    rec_classwise = dict(zip(id2label, [0]*len(id2label)))
    f1_classwise = dict(zip(id2label, [0]*len(id2label)))
    span_prec_classwise = dict(zip(id2label, [0]*len(id2label)))
    span_rec_classwise = dict(zip(id2label, [0]*len(id2label)))
    span_f1_classwise = dict(zip(id2label, [0]*len(id2label)))
    type_acc_classwise = dict(zip(id2label, [0]*len(id2label)))
    print("\n===== Classwise evaluation =====")
    print("Label\tCount\tRate\tLen_min\tLen_max\tLen_mean\tPrec\tRec\tF1")
    for i, k in enumerate(id2label):
        # both span and type correct
        prec_classwise[k] = round(n_correct_classwise[k] / n_pred_classwise[k] * 100, 2) if n_pred_classwise[k] else 0
        rec_classwise[k] = round(n_correct_classwise[k] / n_gold_classwise[k] * 100, 2) if n_gold_classwise[k] else 0
        f1_classwise[k] = round(2 * prec_classwise[k] * rec_classwise[k] / (prec_classwise[k] + rec_classwise[k]), 2) if n_correct_classwise[k] else 0    

        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            k, 
            type2count[k] if k in type2count else 0,
            type2rate[k] if k in type2rate else 0,
            type2lmin[k] if k in type2rate else 0,
            type2lmax[k] if k in type2rate else 0,
            type2lmean[k] if k in type2rate else 0,
            prec_classwise[k], 
            rec_classwise[k], 
            f1_classwise[k]
            ))
        metrics_table.loc[i] = [
            k, 
            type2count[k] if k in type2count else 0,
            type2rate[k] if k in type2rate else 0,
            type2lmin[k] if k in type2rate else 0,
            type2lmax[k] if k in type2rate else 0,
            type2lmean[k] if k in type2rate else 0,
            prec_classwise[k], 
            rec_classwise[k], 
            f1_classwise[k]
            ]
        
    print("TOTAL\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        sum(type2count.values()),
        1,
        tot_lmin,
        tot_lmax,
        tot_lmean,
        prec, 
        rec, 
        f1
        ))
    
    lname2abb = {v:k for k,v in args.abb2lname.items()}
    metrics_table.loc[len(lname2abb)] = [
        "Total", 
        sum(type2count.values()),
        1, 
        tot_lmin,
        tot_lmax,
        tot_lmean,
        prec, 
        rec, 
        f1
        ]
    
    # 写入metric文件
    metrics_table.to_csv(args.metric_path, index=False)

    # ================================= two-stage (span extraction & span classification) metrics ===============================
    print("\n===== Two-stage evaluation =====")
    print("Span-Prec\tSpan-Rec\tSpan-F1\tType-Acc\tPrec\tRec\tF1")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
        span_prec,
        span_rec,
        span_f1,
        type_acc,
        prec, 
        rec, 
        f1
        ))
    metrics_table_twostage = pd.DataFrame(columns=['Span-Prec', 'Span-Rec', 'Span-F1', 'Type-Acc', 'Prec', 'Rec', 'F1'])
    metrics_table_twostage.loc[0] = [
        span_prec,
        span_rec,
        span_f1,
        type_acc,
        prec, 
        rec, 
        f1
        ]

    # 写入twostage metric文件
    # metrics_table_twostage.to_csv(args.twostage_path, index=False)

'''
def parse_response(args, data):
    # SC投票方式
    MV_func = args.MV_func

    data_response_parsed = []
    for i_item, item in enumerate(tqdm(data, desc="parse responses")):
        if args.consistency == 0:
            response = item["response"]
            prediction = response_2_prediction(args, item, response)
            item["prediction"] = prediction
        else:
            responses = item["responses"]
            if isinstance(responses, str):
                responses = eval(responses)
            assert isinstance(responses, list)
            prediction_per_consist = [response_2_prediction(args, item, tmp_resp) for tmp_resp in responses]
            item["prediction_per_consist"] = prediction_per_consist

            # if args.consistency_selection == "two_stage_majority_voting":
            #     prediction = two_stage_majority_voting(args, prediction_per_consist)
            # elif args.consistency_selection == "majority_voting":
            #     prediction = majority_voting(args, prediction_per_consist)
            prediction = MV_func(args, prediction_per_consist)
            item["prediction"] = prediction
            # 计算voted SC答案的score
            consistency_score_entities = compute_consistency_score(prediction_per_consist, voted_prediction=prediction)
            if len(consistency_score_entities):
                consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
            else:
                consistency_score_avg = 0
            item["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg} # 最终投票出的答案中每个实体的consistency score (dict)
            # 计算所有SC答案的score
            consistency_score_SC_all_ans = compute_consistency_score(prediction_per_consist, voted_prediction=None)
            if len(consistency_score_SC_all_ans):
                consistency_score_SC_all_ans_avg = sum(list(consistency_score_SC_all_ans.values())) / len(consistency_score_SC_all_ans)
            else:
                consistency_score_SC_all_ans_avg = 0
            item["consistency_score_SC_all_ans"] = {"entities": consistency_score_SC_all_ans, "avg":consistency_score_SC_all_ans_avg}

        data_response_parsed.append(item)
    
    return data_response_parsed
'''

'''
def collect_all_SC_answers(args, data):
    assert args.consistency == 1

    copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_consist", "prediction"]

    data_w_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="collect all SC answers")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)

        prediction_per_consist = [response_2_prediction(args, item, tmp_resp) for tmp_resp in responses]

        cnt_prediction_tuple = {}
        for tmp_pred in prediction_per_consist:
            for k,v in tmp_pred.items():
                if (k,v) not in cnt_prediction_tuple:
                    cnt_prediction_tuple[(k,v)] = 0
                cnt_prediction_tuple[(k,v)] += 1

        # 将(mention, type)频次转为json格式
        label = item["label"]
        if isinstance(label, str):
            label = eval(label)
        if isinstance(label, list):
            label = json2dict(label)
        prediction_all_with_cnt = []
        prediction_correct_with_cnt = []
        prediction_wrong_with_cnt = []
        for (k,v), cnt in cnt_prediction_tuple.items():
            prediction_all_with_cnt.append(str({k:v, "SC Count": cnt}))
            if k in label and label[k]==v:
                prediction_correct_with_cnt.append(str({k:v, "SC Count": cnt}))
            else:
                prediction_wrong_with_cnt.append(str({k:v, "SC Count": cnt}))

        item_w_SC_all_ans = {}
        for k in copying_keys:
            item_w_SC_all_ans[k] = item[k]

        item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
        item_w_SC_all_ans["SC_correct_ans"] = prediction_correct_with_cnt
        item_w_SC_all_ans["SC_wrong_ans"] = prediction_wrong_with_cnt
        data_w_all_SC_ans.append(item_w_SC_all_ans)
    
    return data_w_all_SC_ans
'''


def main(args):
    # 判断是否需要复制response文件
    if not os.path.exists(args.response_path):
        if args.consistency_selection == "two_stage_majority_voting":
            copying_file_path = args.response_MV_path
        elif args.consistency_selection == "majority_voting":
            copying_file_path = args.response_TSMV_path
        else:
            raise ValueError(f"Unrecognized consistency selection: {args.consistency_selection}")
        copying_data = load_data(copying_file_path)
        copy_file_to_path(copying_data, args.response_dir, args.response_path)
        logger.info(f"File is copied to: {args.response_path}")

    # 加载数据
    data_response = load_data(args.response_path)

    # 是否需要解析回答（暂时只在complete_dialogue形式下实现）
    if args.parse_response == 1:
        data_response, data_w_all_SC_ans = parse_response_std(args, data_response)
        # 输出SC所有的答案（达到阈值 + 未达到阈值的）
        if args.output_SC_all_answer == 1:
            assert len(data_w_all_SC_ans) > 0
            save_data(args.SC_all_ans_path, data_w_all_SC_ans)
            logger.info(f"Data with ALL SC answers saved to: {args.SC_all_ans_path}")  

    # 计算metrics
    compute_metrics(args, data_response)
    
    # 存文件，将prediction dict转为dict,减少占行
    for i in range(len(data_response)):
        data_response[i]["prediction"] = str(data_response[i]["prediction"])
        if args.consistency:
            data_response[i]["responses"] = str(data_response[i]["responses"])
            data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
            data_response[i]["consistency_score_SC_all_ans"] = str(data_response[i]["consistency_score_SC_all_ans"])

    save_data(args.pred_path, data_response)
    logger.info(f"Prediction data saved to: {args.pred_path}")

    # 输出SC所有的答案（达到阈值 + 未达到阈值的)
    # if args.output_SC_all_answer == 1:
    #     data_w_all_SC_ans = collect_all_SC_answers(args, data_response)
    #     save_data(args.SC_all_ans_path, data_w_all_SC_ans)
    #     logger.info(f"Data with ALL SC answers saved to: {args.SC_all_ans_path}")  


def get_paths(args):
    dataname = args.dataname
    if args.dataname == "ace04en":
        dataname = f"{args.dataname}/{args.folder}"

    # 标签集路径 + 加载
    args.abb2labelname_path = f"OPENAI/data/{args.task}/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # 数据集统计信息加载路径
    args.class_n_r_path = f"OPENAI/data/{args.task}/{dataname}/span_classwise_statistics.csv"

    # response加载路径
    folder_0 = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder_0 = f"fs_{folder_0}"    
    if args.few_shot_setting in ["fixed", "pool"]:
        folder_0 = f"{folder_0}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder_0 = f"{folder_0}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder_0 = f"{folder_0}_tool"
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"
        # 备用response路径，用于复制文件
        folder_MV = f"{folder_0}_consist_{args.temperature}_{args.query_times}_MV"
        folder_TSMV = f"{folder_0}_consist_{args.temperature}_{args.query_times}_TSMV"

        # SC投票方式
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting,
                            "majority_voting": majority_voting}
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
    # if args.dataname == "conll2003":
    #     datamode = "conllpp_test"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    # 存储所有SC答案
    SC_ans_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_SC_all_ans.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ParseAns.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    model_folder = model_abb_map[args.model]
    parse_tool_folder = args.parse_tool

    args.response_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{response_filename}"
    args.response_dir = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}"
    args.pred_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{pred_filename}"
    # 存储所有SC答案
    args.SC_all_ans_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{SC_ans_filename}"
    args.metric_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{metric_filename}"
    args.twostage_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{twostage_filename}"
    # if args.parse_tool != "hanlp":
    #     args.response_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{response_filename}"
    #     args.pred_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{pred_filename}"
    #     # 存储所有SC答案
    #     args.SC_all_ans_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{SC_ans_filename}"
    #     args.metric_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{metric_filename}"
    #     args.twostage_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{twostage_filename}"
    
    # 备用response路径，用于计算不同majority voting方法时复制response文件
    if args.consistency==1:
        args.response_MV_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder_MV}/{response_filename}"
        args.response_TSMV_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder_TSMV}/{response_filename}"

    # Logger setting
    log_dir = f"OPENAI/log/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}"
    # if args.parse_tool != "hanlp":
    #     log_dir = f"OPENAI/log/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}"
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
    # [None, key_noun, key_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=1, type=int)
    # choices=["random_42", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos", "SBERTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    # SC筛选答案的方法: [two_stage_majority_voting, ]
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # 输出所有预测答案组，包括高于consistency阈值和未达consistency阈值的
    parser.add_argument("--output_SC_all_answer", default=0, type=int, choices=[0,1])

    # 是否需要将response解析为prediction
    parser.add_argument("--parse_response", default=0, type=int, choices=[0,1])

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1)

    # 解析工具选择
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    # 实验
    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()

    
    assert args.start_time is not None
    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
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

    logger.info("\n\n\n---------- Parse answers ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)