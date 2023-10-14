import json
import logging, logging.config
import os
import pandas as pd

from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

from os import path
import sys
# 导入parent目录下的模块
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import get_logger, load_data, save_data, json2dict, dict2json, assert_gpt35_turbo_16k, run_llm, format_json2str, copy_file_to_path
from utils_parse_answer import response_2_prediction, collect_mention2labels, combine_question_predictions, two_stage_majority_voting, majority_voting, compute_consistency_score, combine_consistency_scores, collect_all_SC_answers, parse_response_qa

from const import dataset_language_map, dataset_label_order_map, my_api_keys, model_abb_map

logger = logging.getLogger()


def compute_metrics(args, data_responses):
    class_n_r = pd.read_csv(args.class_n_r_path)
    mode = args.datamode
    # if args.dataname == "conll2003":
    #     mode = "conllpp_test"
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

'''
def construct_messages_complete_dialogue(args, prompt,responded_questions, responded_answers):
    messages = []
    question = prompt["task_description"]+"\n\n"+prompt["demos_prompts"]+"\n\n"+prompt["query_information"]
    messages.append({"role": "user", "content": question})
    responded_qas = responded_questions[0] + "\n" + responded_answers[0]
    for q, a in zip(responded_questions[1:], responded_answers[1:]):
        responded_qas += "\n" + q + "\n" + a
    messages.append({"role": "assistant", "content": responded_qas})

    return messages

def construct_messages_complete_question(args, prompt,prediction_per_quest):
    messages = []
    output_constraint_json = prompt["output_constraint_json"]
    for i_quest, (quest, ans) in enumerate(zip(prompt["questions"], prediction_per_quest)):
        if i_quest == 0:
            curr_question = prompt["task_description"]
            # 输出格式限制curr_question
            if args.output_constraint_pos == "behindT":
                curr_question += output_constraint_json
            if len(prompt["demos_prompts"]) > 0:
                curr_question += "\n\n" + prompt["demos_prompts"]
            curr_question += "\n\n" + prompt["query_information"]
            curr_question += "\n" + quest
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint_json
        else:
            curr_question = quest
            if args.output_constraint_pos == "behindQ":
                curr_question += output_constraint_json
        messages.append({"role": "user", "content": curr_question})
        messages.append({"role": "assistant", "content": format_json2str(dict2json(ans))})
    return messages

def label_disambiguation(args, prediction_all_json, messages, responses):
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

    return prediction_aggregated, messages, responses

def parse_response(args, data):
    # SC投票方式
    MV_func = args.MV_func

    if args.order is not None:
        # if args.output_constraint_pos != "behindQ": # 对simplified DiaCDia未实现
        #     raise ValueError(f"Not implemented for output_constraint_pos={args.output_constraint_pos}")
        if args.complete_dialogue == 1:
            raise ValueError(f"Not implemented for completing dialogue mode. (complete_dialogue={args.complete_dialogue})")
    else:
        raise ValueError(f"Not implemented for standard setting. (order={args.order})")
    if args.demo_form != "dialogue": # 对simple demo未实现
        raise ValueError(f"Not implemented for demo_form={args.demo_form})")
    
    # 加载prompt
    data_prompts = load_data(args.prompt_path)

    data_response_parsed = []
    for i_item, item in enumerate(tqdm(data, desc="parse responses")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)
        prompt = data_prompts[i_item]["prompt"]

        if args.complete_dialogue == 1: # 补全整个dialogue形式
            assert args.few_shot_setting != "zs"
            assert args.consistency==0 # complete dialogue没有实现consistency
            if len(responses)>1:
                print(f"#Responses more than one: \n{responses}")
            response = responses[0]
            parsed_results = response_2_prediction(args, item, response, complete_form="dialogue", return_responded_qa=True)
            prediction_per_quest, responded_questions, responded_answers = parsed_results

            # 构建对话列表
            messages = construct_messages_complete_dialogue(prompt, responded_questions, responded_answers)

            # lael消歧
            prediction_all_json = []
            for tmp_pred in prediction_per_quest:
                prediction_all_json.extend(dict2json(tmp_pred))

            if len(prediction_all_json)==0:
                prediction_aggregated = {}
            else:
                if args.label_disambiguation == 1:
                    prediction_aggregated, messages, responses = label_disambiguation(args, prediction_all_json, messages, [responses])
                else:
                    prediction_aggregated = combine_question_predictions(args, prediction_per_quest)

            item["prediction_per_quest"] = prediction_per_quest
            item["prediction"] = prediction_aggregated
            
        else: # 普通qa形式
            if args.consistency==0:
                assert len(responses) == len(args.label_order)
                prediction_per_quest = []
                for tmp_resp, tmp_quest in zip(responses, prompt["questions"]):
                    tmp_pred = response_2_prediction(args, item, tmp_resp, question=tmp_quest)
                    prediction_per_quest.append(tmp_pred)
                # 构建对话列表
                messages = construct_messages_complete_question(prompt,prediction_per_quest)

                # lael消歧
                prediction_all_json = []
                for tmp_pred in prediction_per_quest:
                    prediction_all_json.extend(dict2json(tmp_pred))

                if len(prediction_all_json)==0:
                    prediction_aggregated = {}
                else:
                    if args.label_disambiguation == 1:
                        prediction_aggregated, messages, responses = label_disambiguation(args, prediction_all_json, messages, responses)
                    else:
                        prediction_aggregated = combine_question_predictions(args, prediction_per_quest)

                item["prediction_per_quest"] = prediction_per_quest
                item["prediction"] = prediction_aggregated

            else:
                if args.consis_level=="question":
                    # assert len(responses) == len(args.label_order) # 可能是因为label消歧过程
                    assert len(responses[0]) == args.query_times
                    prediction_per_quest = []
                    consistency_score_per_quest = []
                    # SC所有出现过的答案的频次
                    cnt_prediction_tuple = {}
                    for tmp_quest_resps, tmp_quest in zip(responses, prompt["questions"]):
                        tmp_quest_preds = [response_2_prediction(args, item, x, question=tmp_quest) for x in tmp_quest_resps]
                        
                        tmp_voted_pred = MV_func(args, tmp_quest_preds)
                        prediction_per_quest.append(tmp_voted_pred)

                        tmp_consistency_score = compute_consistency_score(tmp_quest_preds, tmp_voted_pred)
                        consistency_score_per_quest.append(tmp_consistency_score)

                        # 统计所有预测中出现的tuple频次
                        for tmp_pred in tmp_quest_preds:
                            for k,v in tmp_pred.items():
                                if (k,v) not in cnt_prediction_tuple:
                                    cnt_prediction_tuple[(k,v)] = 0
                                cnt_prediction_tuple[(k,v)] += 1

                    # 构建对话列表
                    messages = construct_messages_complete_question(prompt,prediction_per_quest)

                    # lael消歧
                    prediction_all_json = []
                    for tmp_pred in prediction_per_quest:
                        prediction_all_json.extend(dict2json(tmp_pred))

                    if len(prediction_all_json)==0:
                        prediction_aggregated = {}
                    else:
                        if args.label_disambiguation == 1:
                            prediction_aggregated, messages, responses = label_disambiguation(args, prediction_all_json, messages, responses)
                        else:
                            prediction_aggregated = combine_question_predictions(args, prediction_per_quest)
                    
                    consistency_score_entities = combine_consistency_scores(consistency_score_per_quest, prediction_aggregated)
                    if len(consistency_score_entities) > 0:
                        consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
                    else:
                        consistency_score_avg = 0

                    item["prediction_per_quest"] = prediction_per_quest
                    item["prediction"] = prediction_aggregated
                    item["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg}
                    # 所有SC答案的score
                    if len(cnt_prediction_tuple):
                        avg_cnt_prediction_tuple = sum(list(cnt_prediction_tuple.values())) / len(cnt_prediction_tuple)
                    else:
                        avg_cnt_prediction_tuple = 0
                    item["consistency_score_SC_all_ans"] = {"entities":cnt_prediction_tuple, "avg":avg_cnt_prediction_tuple}

                elif args.consis_level=="sample":
                    # assert len(responses[0]) == len(args.label_order)
                    assert len(responses) == args.query_times
                    prediction_per_quest_all_consis = []
                    prediction_aggregated_all_consis = []
                    # SC所有出现过的答案的频次
                    cnt_prediction_tuple = {}
                    for i_consis in range(args.query_times):
                        assert len(responses[i_consis]) >= len(prompt["questions"])
                        prediction_per_quest_curr_consis = []
                        for tmp_resp, tmp_quest in zip(responses[i_consis], prompt["questions"]):
                            tmp_pred = response_2_prediction(args, item, tmp_resp, question=tmp_quest)
                            prediction_per_quest_curr_consis.append(tmp_pred)

                        prediction_per_quest_all_consis.append(prediction_per_quest_curr_consis)

                        # 构建对话列表
                        messages_curr_consis = construct_messages_complete_question(prompt,prediction_per_quest_curr_consis)

                        # label消歧
                        prediction_all_json_curr_consis = []
                        for tmp_pred in prediction_per_quest_curr_consis:
                            prediction_all_json_curr_consis.extend(dict2json(tmp_pred))
                        
                        if len(prediction_all_json_curr_consis)==0:
                            prediction_aggregated_curr_consis = {}
                        else:
                            if args.label_disambiguation == 1:
                                prediction_aggregated_curr_consis, messages_curr_consis, responses[i_consis] = label_disambiguation(args, prediction_all_json_curr_consis, messages_curr_consis, responses[i_consis])
                            else:
                                prediction_aggregated_curr_consis = combine_question_predictions(args, prediction_per_quest_curr_consis)
                        prediction_aggregated_all_consis.append(prediction_aggregated_curr_consis)

                        # 统计所有预测过的（mention, type）tuple频次
                        for k,v in prediction_aggregated_curr_consis.items():
                            if (k,v) not in cnt_prediction_tuple:
                                cnt_prediction_tuple[(k,v)] = 0
                            cnt_prediction_tuple[(k,v)] += 1

                    prediction_aggregated = MV_func(args,prediction_aggregated_all_consis)
                    consistency_score_entities=compute_consistency_score(prediction_aggregated_all_consis, prediction_aggregated)
                    if len(consistency_score_entities) > 0:
                        consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
                    else:
                        consistency_score_avg = 0
                    
                    item["prediction_per_quest"] = prediction_per_quest_all_consis
                    item["prediction_per_consist"] = prediction_aggregated_all_consis
                    item["prediction"] = prediction_aggregated
                    item["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg}
                    # 所有SC答案的score
                    if len(cnt_prediction_tuple):
                        avg_cnt_prediction_tuple = sum(list(cnt_prediction_tuple.values())) / len(cnt_prediction_tuple)
                    else:
                        avg_cnt_prediction_tuple = 0
                    item["consistency_score_SC_all_ans"] = {"entities":cnt_prediction_tuple, "avg":avg_cnt_prediction_tuple}
                else:
                    raise f"Unknown consis_level={args.consis_level}"

        data_response_parsed.append(item)

    return data_response_parsed
'''

'''
def collect_all_SC_answers_question_level(args, data):
    id2label = args.id2label

    copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]

    data_w_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="collect all SC answers")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)
        
        # 处理每个question的多次response
        # 收集所有与(mention, type)对的频次
        cnt_prediction_tuple = {}
        for i_label, label in enumerate(id2label):
            curr_label_resps = responses[i_label]
            assert isinstance(curr_label_resps, list)
            for tmp_resp in curr_label_resps:
                tmp_pred = response_2_prediction(args, item, tmp_resp, question=label)
                for k, v in tmp_pred.items():
                    if (k,v) not in cnt_prediction_tuple:
                        cnt_prediction_tuple[(k,v)] = 0
                    cnt_prediction_tuple[(k,v)] += 1

        # 将(mention, type)频次转为json格式
        prediction_all_with_cnt = []
        for (k,v), cnt in cnt_prediction_tuple.items():
            prediction_all_with_cnt.append(str({
                k:v,
                "SC Count": cnt
            }))
        
        item_w_SC_all_ans = {}
        for k in copying_keys:
            item_w_SC_all_ans[k] = item[k]

        item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
        data_w_all_SC_ans.append(item_w_SC_all_ans)

    return data_w_all_SC_ans

def collect_all_SC_answers_sample_level(args, data):
    id2label = args.id2label

    copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]

    data_w_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="collect all SC answers")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)
        
        # 处理每个question的多次response
        # 收集所有与(mention, type)对的频次
        cnt_prediction_tuple = {}
        for i_consis, consis in enumerate(range(args.query_times)):
            curr_consis_resps = responses[i_consis]
            assert isinstance(curr_consis_resps, list)
            for i_label, label in enumerate(id2label):
                curr_label_resp = curr_consis_resps[i_label]
                curr_label_pred = response_2_prediction(args, item, curr_label_resp, question=label)
                for k, v in curr_label_pred.items():
                    if (k,v) not in cnt_prediction_tuple:
                        cnt_prediction_tuple[(k,v)] = 0
                    cnt_prediction_tuple[(k,v)] += 1

        # 将(mention, type)频次转为json格式
        prediction_all_with_cnt = []
        for (k,v), cnt in cnt_prediction_tuple.items():
            prediction_all_with_cnt.append(str({
                k:v,
                "SC Count": cnt
            }))
        
        item_w_SC_all_ans = {}
        for k in copying_keys:
            item_w_SC_all_ans[k] = item[k]

        item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
        data_w_all_SC_ans.append(item_w_SC_all_ans)

    return data_w_all_SC_ans
'''

'''
def collect_all_SC_answers(args, data):
    assert args.consistency == 1
    id2label = args.id2label

    copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]

    data_w_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="collect all SC answers")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)
        
        # 收集所有与(mention, type)对的频次
        cnt_prediction_tuple = {}
        if args.consis_level == "question":
            for i_label, label in enumerate(id2label):
                curr_label_resps = responses[i_label]
                assert isinstance(curr_label_resps, list)
                for tmp_resp in curr_label_resps:
                    tmp_pred = response_2_prediction(args, item, tmp_resp, question=label)
                    for k, v in tmp_pred.items():
                        if (k,v) not in cnt_prediction_tuple:
                            cnt_prediction_tuple[(k,v)] = 0
                        cnt_prediction_tuple[(k,v)] += 1
        elif args.consis_level == "sample":
            for i_consis, consis in enumerate(range(args.query_times)):
                curr_consis_resps = responses[i_consis]
                assert isinstance(curr_consis_resps, list)
                for i_label, label in enumerate(id2label):
                    curr_label_resp = curr_consis_resps[i_label]
                    curr_label_pred = response_2_prediction(args, item, curr_label_resp, question=label)
                    for k, v in curr_label_pred.items():
                        if (k,v) not in cnt_prediction_tuple:
                            cnt_prediction_tuple[(k,v)] = 0
                        cnt_prediction_tuple[(k,v)] += 1
        else:
            raise ValueError(f"UNrecognized consistency level: {args.consis_level}")
        
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
        data_response, data_w_all_SC_ans = parse_response_qa(args, data_response)
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
        data_response[i]["responses"] = str(data_response[i]["responses"])
        if args.consistency:
            if args.consis_level == "question":
                data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])
            elif args.consis_level == "sample":
                data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])
                data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
            data_response[i]["consistency_score_SC_all_ans"] = str(data_response[i]["consistency_score_SC_all_ans"])
        else:
            data_response[i]["prediction_per_quest"] = str(data_response[i]["prediction_per_quest"])

    save_data(args.pred_path, data_response)
    logger.info(f"Prediction data saved to: {args.pred_path}")    

    # 输出SC所有的答案（达到阈值 + 未达到阈值的）
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
    prompt_folder = folder_0
    if args.label_disambiguation==1:
        folder_0 += "_LD"
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_{flag_majority_voting}"
        # 备用response路径，用于复制文件
        folder_MV = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_MV"
        folder_TSMV = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_TSMV"

        # SC投票方式
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting,
                            "majority_voting": majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]
    else:
        folder=folder_0

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
    if args.consistency==1:
        folder_MV = f"{qa_tag}_{args.order}_{folder_MV}"
        folder_TSMV = f"{qa_tag}_{args.order}_{folder_TSMV}"
    prompt_folder = f"{qa_tag}_{args.order}_{prompt_folder}"

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
    prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    # 存储所有SC答案
    SC_ans_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_SC_all_ans.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ParseAns.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    model_folder = model_abb_map[args.model]
    parse_tool_folder = args.parse_tool

    args.prompt_path = f"OPENAI/prompts/{args.task}/{parse_tool_folder}/{dataname}/{prompt_folder}/{prompt_filename}"
    args.response_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{response_filename}"
    args.response_dir = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}"
    args.pred_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{pred_filename}"
    # 存储所有SC答案
    args.SC_all_ans_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{SC_ans_filename}"
    args.metric_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{metric_filename}"
    args.twostage_path = f"OPENAI/result/{args.task}/{model_folder}/{parse_tool_folder}/{dataname}/{folder}/{twostage_filename}"
    # if args.parse_tool != "hanlp":
    #     args.prompt_path = f"OPENAI/prompts/{args.task}/{args.parse_tool}/{dataname}/{prompt_folder}/{prompt_filename}"
    #     args.response_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{response_filename}"
    #     # 存储所有SC答案
    #     args.SC_all_ans_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{SC_ans_filename}"
    #     args.pred_path = f"OPENAI/result/{args.task}/{model_folder}/{args.parse_tool}/{dataname}/{folder}/{pred_filename}"
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
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
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
    parser.add_argument("--demo_select_method", default="GPTEmbClusterKmeans") # , choices=["random", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)
    # qa模式下，demo的形式：(1) 整段conversation; (2) 每个类别的几个entity例子。
    parser.add_argument("--demo_form", default="dialogue", type=str, choices=["dialogue", "entity"])    
    # qa few-shot模式下，两种补全形式：(1)和zero-shot一致的问答；(2)补全整个对话
    parser.add_argument("--complete_dialogue", default=0, type=int, choices=[0, 1])
    # output constraint的位置
    parser.add_argument("--output_constraint_pos", default="behindQ", type=str, choices=["behindT", "behindQ"])
    parser.add_argument("--OC1", default=0, type=int, choices=[0,1])
    # 是否进行label消歧
    parser.add_argument("--label_disambiguation", default=0, type=int, choices=[0, 1])

    # QA setting    
    parser.add_argument("--order", default=None, type=str) # chatgpt0, chatgpt1, chatgpt2

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int, choices=[0,1])
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    # SC筛选答案的方法: [two_stage_majority_voting, majority_voting]
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

    # 设置prompt_pool，用以在AskGPT过程中动态生成所需prompt
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

    if args.output_SC_all_answer == 1:
        assert args.parse_response == 1

    assert_gpt35_turbo_16k(args, chat_paradigm="qa_dialogue")     

    args = get_paths(args)

    args.api_key = my_api_keys

    if not ("gpt" in args.model) and not ("text" in args.model):
        args.api_key = "EMPTY"

    logger.info("\n\n\n---------- Parse answers ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)