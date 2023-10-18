import logging, logging.config
from typing import List, Dict
from argparse import Namespace
import re
from utils import json2dict

logger = logging.getLogger()

def response_2_prediction_of_list(args, query, response, resp_idx=None, question=None):
    '''
    Returns: 
        predictions: list
    '''
    if response in ["", "[]", "[{}]"]:
        return []

    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]
    label_order =args.label_order
    target_type = label_order[resp_idx]
    if isinstance(target_type, list):
        if len(target_type) > 1:
            raise ValueError(f"target type is more than one: {len(target_type)}")
        target_type = target_type[0]

    # 把中文标点替换成英文标点
    punc_zh2en = {'，': ',', '。': '.', '：': ':'}
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    # 去掉所有换行符号
    response_punctransed = response_punctransed.replace("\n", "")
    # 匹配符合答案格式的模式
    matched_list = re.findall(r'\[(.*?)\]', response_punctransed)
    if len(matched_list) == 0:
        # 如果匹配不到list格式
        logger.info(f"===== Error occured (Wrong Format): {sentid}")
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info("        Question: {}".format(question))
        logger.info(f"        Error response_{resp_idx}: \n{response}")
        logger.info("        Set and_processed as empty dict.")
        prediction = []
    else:
        try:
            # 选择匹配到的最后一个字符串。因为推理的最后一步是输出最终认为正确的答案。                
            ans_str = matched_list[-1]
            if "\"" in ans_str:
                ans_str = "[" + ans_str + "]"
                prediction = eval(ans_str)
            elif ans_str == "":
                prediction = []
            else:
                # 直接以 "," 分割
                ans_ls_raw = ans_str.split(",")
                # 去掉可能有的空格
                prediction = [x.strip() for x in ans_ls_raw]
            
        except Exception as e:
            logger.info(f"===== Error occured (Wrong Format): {sentid}")
            logger.info("        Sent: {}".format(sent))
            logger.info("        Label: {}".format(label))
            logger.info("        Question: {}".format(question))
            logger.info(f"        Error response_{resp_idx}: \n{response}")
            logger.info("        Set and_processed as empty dict.")
            logger.info(f"        Error traceback:")
            logger.info(str(e))    
            prediction = []
    
    return prediction    

def response_2_prediction_of_dict_json(args, query, response, resp_idx=None, question=None, return_form="dict"):
    # if return empty answer
    if response in ["", "[]", "[{}]", "A: []", "{}"]:
        prediction = [] if return_form=="json" else {}
        return prediction

    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]
    id2label =args.id2label

    # 把中文标点替换成英文标点
    punc_zh2en = {'，': ',', '。': '.', '：': ':'}
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    # 去掉所有换行符号
    response_punctransed = response_punctransed.replace("\n", "")
    # 匹配符合答案格式的模式
    matched_list = re.findall(r'\[(.*?)\]', response_punctransed)
    if len(matched_list) == 0:
        # 如果匹配不到json格式
        # 首先匹配dict格式
        if args.few_shot_setting == "zs":
            matched_list = re.findall(r'\{(.*?)\}', response_punctransed)
            prediction = []
            for matched_item in matched_list:
                matched_item = "{" + matched_item + "}"
                # 将null替换成\"O\"
                matched_item = matched_item.replace("null", "\"O\"")
                eval_matched_item = eval(matched_item)
                if isinstance(eval_matched_item, dict):
                    for k, v in eval_matched_item.items():
                        if k in sent and v in id2label:
                            prediction.append({k:v})

            if len(prediction)>0:
                if return_form=="dict":
                    prediction=json2dict(prediction)
                return prediction


        logger.info(f"===== Error occured (No matched): {sentid}")
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info("        Question: {}".format(question))
        logger.info(f"        Error response_{resp_idx}: \n{response}")
        logger.info("        Set and processed as empty dict.")
        prediction = []
    else:
        try:
            # 在匹配的字符串两端加上列表括号[]
            # 选择匹配到的最后一个字符串。因为推理的最后一步是输出最终认为正确的答案。                
            ans_str = '[' + matched_list[-1] + ']'
            # 将null替换成\"O\"
            ans_str = ans_str.replace("null", "\"O\"")
            
            ans_eval = eval(ans_str)

            # 如果返回的是空列表
            if len(ans_eval)==0:
                prediction = ans_eval
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction

            # 处理如下格式：[{"Entity Name": "Oleg Shatskiku", "Entity Label": "PERSON"}, ...]
            if "Entity Name" in ans_eval[0] and "Entity Label" in ans_eval[0]:
                prediction = []
                for tmp in ans_eval:
                    if len(tmp) == 0:
                        continue
                    if tmp["Entity Name"] in id2label: #将两个value的值反过来了
                        tmp_ment = tmp["Entity Label"]
                        tmp_type = tmp["Entity Name"]
                    else:
                        tmp_ment = tmp["Entity Name"] 
                        tmp_type = tmp["Entity Label"] 
                    prediction.append({tmp_ment:tmp_type})
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction
            
            if "实体名称" in ans_eval[0] and "实体标签" in ans_eval[0]:
                prediction = []
                for tmp in ans_eval:
                    if tmp["实体名称"] in id2label: #将两个value的值反过来了
                        tmp_ment = tmp["实体标签"] 
                        tmp_type = tmp["实体名称"]
                    else:
                        tmp_ment = tmp["实体名称"]
                        tmp_type = tmp["实体标签"] 
                    prediction.append({tmp_ment:tmp_type})
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction
            
            # 处理两种可能的有效输出格式：
            # 1： [{XX:XX, XX:XX, XX:XX}]
            # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
            
            if len(ans_eval) == 1 and len(ans_eval[0]) > 1: # 1： [{XX:XX, XX:XX, XX:XX}]
                prediction_w_o = [
                    {k: v} for k,v in ans_eval[0].items()
                ]
            else: # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
                # prediction_w_o = {list(item.keys())[0]: list(item.values())[0] for item in ans_eval}
                prediction_w_o = ans_eval
            # 删掉typing为"O" (null)的答案
            prediction = []
            for item in prediction_w_o:
                k, v = list(item.items())[0]
                if v != "O":
                    prediction.append(item)
        except Exception as e:
            logger.info(f"===== Error occured (Unparsable): {sentid}")
            logger.info("        Sent: {}".format(sent))
            logger.info("        Label: {}".format(label))
            logger.info("        Question: {}".format(question))
            logger.info(f"        Error response_{resp_idx}: \n{response}")
            logger.info("        Set and_processed as empty dict.")
            logger.info(f"        Error traceback:")
            logger.info(str(e))    
            prediction = []
    
    if return_form=="dict":
        prediction=json2dict(prediction)
    return prediction

def response_2_prediction(args, query, response, resp_idx=None, question=None, return_form="dict", complete_form="question", return_responded_qa=False):
    if complete_form == "question": # 补全每个问题的答案
        if return_form in ["dict", "json"]:
            prediction = response_2_prediction_of_dict_json(args, query, response, resp_idx=resp_idx, question=question, return_form=return_form)
        elif return_form == "list":
            prediction = response_2_prediction_of_list(args, query, response, resp_idx=resp_idx, question=question)
        else:
            raise ValueError(f"Unrecognized return_form: {return_form}")
        return prediction
    else:
        raise ValueError(f"Unrecognized complete_form={complete_form}")

def two_stage_majority_voting(args, prediction_ls=None, mention_type_cnt_all=None):
    '''
    Vote for most consistent named entities from a set of predictions.
    Two-stage voting: 1) entity mention; 2) entity type.
    Params:
        prediction_ls: list of prediction (dict);
    Returns:
        prediction_voted: voted prediction (dict)
    '''
    if isinstance(args, Namespace):
        tot_votes = args.query_times
    elif isinstance(args, int):
        tot_votes = args
    else:
        raise TypeError(f"Unknown type of args: {type(args)}")
    lowest_votes_for_O = tot_votes // 2
    if tot_votes % 2 == 1:
        lowest_votes_for_O += 1

    if mention_type_cnt_all is None:
        assert prediction_ls is not None
        mentions_all = []
        types_all = []
        for tmp_pred in prediction_ls:
            # convert json format to dict
            if isinstance(tmp_pred, list):
                tmp_pred = json2dict(tmp_pred)
            mentions_all += list(tmp_pred.keys())
            types_all += list(tmp_pred.values())

        mention_type_cnt_all = {}
        for tmp_mention, tmp_type in zip(mentions_all, types_all):
            if tmp_mention not in mention_type_cnt_all:
                mention_type_cnt_all[tmp_mention] = {}
            if tmp_type not in mention_type_cnt_all[tmp_mention]:
                mention_type_cnt_all[tmp_mention][tmp_type] = 0
            mention_type_cnt_all[tmp_mention][tmp_type] += 1
        
    # mentions_all_cnt = Counter(mentions_all)
    mentions_all_cnt = {}
    for k,v in mention_type_cnt_all.items():
        mentions_all_cnt[k] = sum(list(v.values()))
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


def combine_question_predictions(args, predictions, return_form="dict"):
    prediction = {}
    if return_form in ["dict", "json"]:
        for tmp_preds in predictions:
            if isinstance(tmp_preds, list):
                tmp_preds = json2dict(tmp_preds)
            for k, v in tmp_preds.items():
                prediction[k] = v
    elif return_form == "list":
        label_order = args.label_order
        # 当前只实现 one-by-one
        assert len(label_order[0])==1
        for labels, pred in zip(label_order, predictions):
            if len(pred) == 0:
                continue
            for p in pred:
                prediction[p] = labels[0]
    else:
        raise ValueError(f"Unrecognized return_form={return_form}")
    
    return prediction

def compute_consistency_score(prediction_ls, voted_prediction=None):
    '''
    Vote for most consistent named entities from a set of predictions.
    Params:
        prediction_ls: list of prediction (dict);
        voted_prediction: voted prediction (dict).
    Returns:
        consistency_score: consist_score of voted prediction (dict)
    '''
    # convert json format to dict
    if isinstance(prediction_ls[0], list):
        prediction_ls = [json2dict(x) for x in prediction_ls]
    
    consistency_score_entities = {}
    # 对voted后的结果计算consistency score
    if voted_prediction != None:
        for k, v in voted_prediction.items():
            consistency_score_entities[(k, v)] = 0
            for tmp_prediction in prediction_ls:
                if k in tmp_prediction and v==tmp_prediction[k]:
                    consistency_score_entities[(k, v)] += 1
    # 计算所有预测结果的consistency score
    else:
        for tmp_prediction in prediction_ls:
            for k,v in tmp_prediction.items():
                if (k,v) not in consistency_score_entities:
                    consistency_score_entities[(k,v)] = 0
                consistency_score_entities[(k,v)] += 1
    
    return consistency_score_entities

def combine_consistency_scores(consistency_scores, prediction_agg):
    consistency_score_agg = {}
    for tmp_consistency_score in consistency_scores:
        for k in tmp_consistency_score:
            mention, type = k[0], k[1]
            if mention in prediction_agg and type==prediction_agg[mention]:
                consistency_score_agg[k] = tmp_consistency_score[k]
    
    return consistency_score_agg

def collect_mention2labels(all_predictions: List[Dict]):
    mention2labels = {} # entity name: [entity label 1, entity label 2]
    for pred in all_predictions:
        mention = list(pred.keys())[0]
        type = pred[mention]
        if mention not in mention2labels:
            mention2labels[mention] = []
        mention2labels[mention].append(type)
    
    return mention2labels