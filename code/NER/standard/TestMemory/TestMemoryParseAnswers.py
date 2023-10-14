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
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) )) ) # 往上2层目录

from utils import get_logger, load_data, save_data, json2dict
from utils_parse_answer import response_2_prediction

logger = logging.getLogger()

'''
def response_2_prediction(args, query, response, resp_idx=None):
    if response == "":
        return {}    
    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]
    id2label =args.id2label

    # 把中文标点替换成英文标点
    punc_zh2en = {'，': ',', '。': '.', '：': ':'}
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    # 匹配符合答案格式的模式
    matched_list = re.findall(r'\[(.*?)\]', response_punctransed)
    if len(matched_list) == 0:
        # 如果匹配不到json格式
        # 首先匹配dict格式
        if args.few_shot_setting == "zs":
            matched_list = re.findall(r'\{(.*?)\}', response_punctransed)
            prediction = {}
            for matched_item in matched_list:
                matched_item = "{" + matched_item + "}"
                # 将null替换成\"O\"
                matched_item = matched_item.replace("null", "\"O\"")
                eval_matched_item = eval(matched_item)
                if isinstance(eval_matched_item, dict):
                    for k, v in eval_matched_item.items():
                        if k in sent and v in id2label:
                            prediction[k] = v
            if len(prediction)>0:
                return prediction


        logger.info(f"===== Error occured (Wrong Format): {sentid}")
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info(f"        Error response_{resp_idx}: \n{response}")
        logger.info("        Set and_processed as empty dict.")
        prediction = {}
    else:
        try:
            # 在匹配的字符串两端加上列表括号[]
            # 选择匹配到的最后一个字符串。因为推理的最后一步是输出最终认为正确的答案。                
            ans_str = '[' + matched_list[-1] + ']'
            # 将null替换成\"O\"
            ans_str = ans_str.replace("null", "\"O\"")
            
            # 处理两种可能的有效输出格式：
            # 1： [{XX:XX, XX:XX, XX:XX}]
            # 2： [{XX:XX}, {XX:XX}, {XX:XX}]

            ans_eval = eval(ans_str)
            
            if len(ans_eval) == 1 and len(ans_eval[0]) > 1: # 1： [{XX:XX, XX:XX, XX:XX}]
                prediction_w_o = {
                    k: v for k,v in ans_eval.items()
                }
            else: # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
                prediction_w_o = {
                    list(item.keys())[0]: list(item.values())[0] for item in ans_eval
                }
            # 删掉typing为"O" (null)的答案
            prediction = {}
            for k, v in prediction_w_o.items():
                if v != "O":
                    prediction[k] = v
        except Exception as e:
            logger.info(f"===== Error occured (Wrong Format): {sentid}")
            logger.info("        Sent: {}".format(sent))
            logger.info("        Label: {}".format(label))
            logger.info(f"        Error response_{resp_idx}: \n{response}")
            logger.info("        Set and_processed as empty dict.")
            logger.info(f"        Error traceback:")
            logger.info(str(e))    
            prediction = {}
    
    return prediction
'''

def majority_vote_ner(args, prediction_ls):
    tot_votes = args.query_times
    lowest_votes_for_O = tot_votes // 2 + 1

    mentions_all = []
    types_all = []
    for tmp_ls in prediction_ls:
        mentions_all += list(tmp_ls.keys())
        types_all += list(tmp_ls.values())


    mention_type_cnt_all = {}
    for tmp_mention, tmp_type in zip(mentions_all, types_all):
        if tmp_mention not in mention_type_cnt_all:
            mention_type_cnt_all[tmp_mention] = {}
        if tmp_type not in mention_type_cnt_all[tmp_mention]:
            mention_type_cnt_all[tmp_mention][tmp_type] = 0
        mention_type_cnt_all[tmp_mention][tmp_type] += 1
        

    mentions_all_cnt = Counter(mentions_all)
    voted_mentions = []
    for tmp_mention in mentions_all_cnt:
        if mentions_all_cnt[tmp_mention] >= lowest_votes_for_O:
            voted_mentions.append(tmp_mention)

    prediction_voted = {}
    for tmp_mention in voted_mentions:
        tmp_type_cnt = mention_type_cnt_all[tmp_mention]
        tmp_type_cnt = list(sorted(list(tmp_type_cnt.items()), key=lambda x: x[1], reverse=True))
        tmp_majority_type, tmp_majority_type_votes = tmp_type_cnt[0][0], tmp_type_cnt[0][1]

        prediction_voted[tmp_mention] = tmp_majority_type

    return prediction_voted


def parse_answers_per_query(args, query):
    response = query["response"]
    prediction = response_2_prediction(args, query, response)

    return prediction

def parse_answers_per_query_multiquery(args, query):
    prediction_ls = []
    for i_resp in range(args.query_times):
        tmp_response = query[f"response_{i_resp}"]
        tmp_prediction = response_2_prediction(args, query, tmp_response)
        prediction_ls.append(tmp_prediction)
    
    prediction_voted = majority_vote_ner(args, prediction_ls)

    return prediction_voted



def parse_answers_batch(args, data_responses):
    bar = tqdm(data_responses)
    for i_query, query in enumerate(bar):
        bar.set_description("Parse Answers")

        if args.consistency:
            prediction = parse_answers_per_query_multiquery(args, query)
        else:
            prediction = parse_answers_per_query(args,query)

        data_responses[i_query]["prediction"] = prediction
    
    return data_responses

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




def main(args):
    # 加载数据
    data_response = load_data(args.response_path)

    # 特殊处理
    paths = [
        "OPENAI/result/NER/scierc/zs/06111946_test__0_response.txt",
        "OPENAI/result/NER/scierc/zs/06111956_test_noun_conj_first_b_0_response.txt",
        "OPENAI/result/NER/scierc/zs/06111957_test_pos_conj_first_b_0_response.txt",
    ]
    if args.response_path in paths:
        for i, item in enumerate(data_response):
            label = item["label"]
            for k, v in label.items():
                label[k] = args.abb2lname[v]
            data_response[i]["label"] = label


    # 将ChatGPT回答解析为NER结果
    data_responses = parse_answers_batch(
        args, 
        data_response
    )

    # 计算metrics
    compute_metrics(args, data_responses)
    

    # 存文件，将prediction dict转为dict,减少占行
    for i in range(len(data_responses)):
        data_responses[i]["prediction"] = str(data_responses[i]["prediction"])
    save_data(args.pred_path, data_responses)



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
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder = f"fs_{folder}"    
    if args.few_shot_setting in ["fixed", "pool"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder = f"{folder}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder = f"{folder}_tool"
    # test memory标记
    folder = f"{folder}_testmem_{args.memory_selection}_{args.memory_shot}"
    # consistency标记    
    if args.consistency:
        folder = f"{folder}_consist_{args.temperature}_{args.query_times}"    

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    datamode = args.datamode
    if args.dataname == "conll2003":
        datamode = "conllpp_test"        
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ParseAns.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    args.response_path = f"OPENAI/result/{args.task}/{dataname}/{folder}/{response_filename}"
    args.pred_path = f"OPENAI/result/{args.task}/{dataname}/{folder}/{pred_filename}"
    args.metric_path = f"OPENAI/result/{args.task}/{dataname}/{folder}/{metric_filename}"
    args.twostage_path = f"OPENAI/result/{args.task}/{dataname}/{folder}/{twostage_filename}"

    # Logger setting
    log_dir = f"OPENAI/log/{args.task}/{dataname}/{folder}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"OPENAI/config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train, test"])
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--max_token_len", default=4096, type=int)
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
    parser.add_argument("--demo_select_method", default="manual1", choices=["random", "GPTEmbClusterKmeans"])
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)

    # tool aug
    parser.add_argument("--tool_aug", default="ToolTokCoarse", choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1)

    # test memory
    parser.add_argument("--memory_selection", default="GPTEmbCos", type=str, choices=["random", "SBertEmbCos", "GPTEmbCos"])
    parser.add_argument("--memory_shot", default=1, type=int)    

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