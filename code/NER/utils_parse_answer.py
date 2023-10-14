import logging, logging.config
from typing import List, Dict
from argparse import Namespace
import re
from collections import Counter
from tqdm import tqdm
from utils import json2dict, dict2json, load_data, run_llm, format_json2str

logger = logging.getLogger()

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


def parse_response_qa(args, data):
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
    data_w_SC_all_ans = []
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
            messages = construct_messages_complete_dialogue(args, prompt, responded_questions, responded_answers)

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
                messages = construct_messages_complete_question(args, prompt,prediction_per_quest)

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
                    messages = construct_messages_complete_question(args, prompt,prediction_per_quest)

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
                        messages_curr_consis = construct_messages_complete_question(args,prompt,prediction_per_quest_curr_consis)

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
                
                # 输出所有SC答案
                if args.output_SC_all_answer:
                    item_w_SC_all_ans = collect_all_SC_answers(args, item, prediction_tuple_count=cnt_prediction_tuple)
                    data_w_SC_all_ans.append(item_w_SC_all_ans)
        data_response_parsed.append(item)

    return data_response_parsed, data_w_SC_all_ans

def parse_response_std(args, data):
    data_response_parsed = []
    data_with_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="parse responses")):
        if args.consistency == 0:
            response = item["response"]
            prediction = response_2_prediction(args, item, response)
            item["prediction"] = prediction
        else:
            # SC投票方式
            MV_func = args.MV_func
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

            # 输出SC所有答案
            if args.output_SC_all_answer==1:
                item_w_all_SC_ans = collect_all_SC_answers(args, item, prediction_tuple_count=consistency_score_SC_all_ans)
                data_with_all_SC_ans.append(item_w_all_SC_ans)

        data_response_parsed.append(item)
    
    return data_response_parsed, data_with_all_SC_ans

def collect_all_SC_answers(args, item, prediction_tuple_count):
    assert args.consistency==1

    if hasattr(args, "order") and args.order != None:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]
    else:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_consist", "prediction"]
    
    # 将(mention, type)频次转为json格式
    label = item["label"]
    if isinstance(label, str):
        label = eval(label)
    if isinstance(label, list):
        label = json2dict(label)
    prediction_voted = item["prediction"]
    if isinstance(prediction_voted, str):
        prediction_voted = eval(prediction_voted)
    if isinstance(prediction_voted, list):
        prediction_voted = json2dict(prediction_voted)

    prediction_all_with_cnt = []
    prediction_correct_with_cnt = []
    prediction_wrong_with_cnt = []
    prediction_voted_with_cnt = []
    for (k,v), cnt in prediction_tuple_count.items():
        prediction_all_with_cnt.append(str({k:v, "SC Count": cnt}))
        if k in label and label[k]==v:
            prediction_correct_with_cnt.append(str({k:v, "SC Count": cnt}))
        else:
            prediction_wrong_with_cnt.append(str({k:v, "SC Count": cnt}))
        if k in prediction_voted and prediction_voted[k]==v:
            prediction_voted_with_cnt.append(str({k:v, "SC Count": cnt}))

    item_w_SC_all_ans = {}
    for k in copying_keys:
        item_w_SC_all_ans[k] = item[k]

    item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
    item_w_SC_all_ans["SC_correct_ans"] = prediction_correct_with_cnt
    item_w_SC_all_ans["SC_wrong_ans"] = prediction_wrong_with_cnt
    item_w_SC_all_ans["SC_voted_ans"] = prediction_voted_with_cnt

    return item_w_SC_all_ans

'''
def collect_all_SC_answers(args, data):
    assert args.consistency==1
    id2label = args.id2label

    if hasattr(args, "order") and args.order != None:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]
    else:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_consist", "prediction"]

    data_w_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="collect all SC answers")):
        responses = item["responses"]
        if isinstance(responses, str):
            responses = eval(responses)
        assert isinstance(responses, list)
        
        cnt_prediction_tuple = {}
        if hasattr(args, "order") and args.order != None:
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
        else:
            prediction_per_consist = [response_2_prediction(args, item, tmp_resp) for tmp_resp in responses]
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
        prediction_voted = item["prediction"]
        if isinstance(prediction_voted, str):
            prediction_voted = eval(prediction_voted)
        if isinstance(prediction_voted, list):
            prediction_voted = json2dict(prediction_voted)

        prediction_all_with_cnt = []
        prediction_correct_with_cnt = []
        prediction_wrong_with_cnt = []
        prediction_voted_with_cnt = []
        for (k,v), cnt in cnt_prediction_tuple.items():
            prediction_all_with_cnt.append(str({k:v, "SC Count": cnt}))
            if k in label and label[k]==v:
                prediction_correct_with_cnt.append(str({k:v, "SC Count": cnt}))
            else:
                prediction_wrong_with_cnt.append(str({k:v, "SC Count": cnt}))
            if k in prediction_voted and prediction_voted[k]==v:
                prediction_voted_with_cnt.append(str({k:v, "SC Count": cnt}))

        item_w_SC_all_ans = {}
        for k in copying_keys:
            item_w_SC_all_ans[k] = item[k]

        item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
        item_w_SC_all_ans["SC_correct_ans"] = prediction_correct_with_cnt
        item_w_SC_all_ans["SC_wrong_ans"] = prediction_wrong_with_cnt
        item_w_SC_all_ans["SC_voted_ans"] = prediction_voted_with_cnt
        data_w_all_SC_ans.append(item_w_SC_all_ans)
    
    return data_w_all_SC_ans
'''

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
    if response in ["", "[]", "[{}]", "A: []"]:
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

def response_of_dialogue_2_prediction(args, query, response, return_form="dict", return_responded_qa=False):
    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]    
    label_order = args.label_order

    # 修正原始的回应形式-0
    response = response.replace('问题：', 'Q: ')
    response = response.replace('Question: ', 'Q: ')
    response = response.replace('答案：', 'A: ')
    response = response.replace('Answer: ', 'A: ')
    response = response.replace('\n\n', '\n')
    if response.endswith('\n'):
        response = response[:-1]
    if response.startswith('\n'):
        response = response[1:]

    responses = response.split("\n")

    if len(responses) == 2*len(label_order)+1:
        if responses[0].startswith("A: "):
            responses = responses[1:]
        elif responses[-1].startswith("Q: "):
            responses = responses[:-1]
        elif not responses[0].startswith("Q: "):
            responses = responses[1:]
        elif not responses[-1].startswith("A: "):
            responses = responses[:-1]
    elif len(responses) == 4*len(label_order)+1:
        responses = responses[:2*len(label_order)]
    elif len(responses) == 2*len(label_order)-1:
        if responses[0].startswith("Q: ") or responses[-1].startswith("Q: "):
            responses.append("A: []")
        elif responses[0].startswith("A: ") or responses[-1].startswith("A: "):
            responses = ["Q: Missed first question."] + responses
    elif len(responses) == 2*len(label_order)-3:
        if response[0].startswith("Q: ") or responses[-1].startswith("Q: "):
            responses.append("A: []")
            responses.append("Q: Missed last question.")
            responses.append("A: []")
    elif len(responses) > 2*len(label_order):
        tmp_responses_questions = [x for i,x in enumerate(responses[:2*len(label_order)]) if i%2==0]
        tmp_responses_answers = [x for i,x in enumerate(responses[:2*len(label_order)]) if i%2==1]
        check_format_quest = [x.startswith("Q: ") for x in tmp_responses_questions]
        check_format_ans = [x.startswith("A: ") for x in tmp_responses_answers]
        # TODO: debug
        # print("Very long list !!!!!!!!!!!!!!!!!!!")
        # print(tmp_responses_questions)
        # print(tmp_responses_answers)
        # print(check_format_quest)
        # print(check_format_ans)

        if sum(check_format_quest)==len(check_format_quest) and sum(check_format_ans)==len(check_format_ans):
            responses = responses[:2*len(label_order)]

    if len(responses) != 2*len(label_order):
        logger.info(f"===== Error occured: len(responses)({len(responses)})!=len(label_order)({len(label_order)})")
        logger.info("        Sentid: {}".format(sentid))
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info(f"       response: \n{responses}")
        logger.info("        Set all answers to empty dict.")
        responses_questions, responses_answers = [], []
        # responses_questions = [[] for i in range(len(label_order))]
        # responses_answers = [[] for i in range(len(label_order))]
        if return_form=="dict":
            predictions = [{} for i in range(len(label_order))]
        elif return_form in ["json", "list"]:
            predictions = [[] for i in range(len(label_order))]
        else:
            raise ValueError(f"Unrecognized return_form={return_form}")

    else:    
        responses_questions = [x for i,x in enumerate(responses) if i%2==0]
        responses_answers = [x for i,x in enumerate(responses) if i%2==1]
        # responses = [x for i,x in enumerate(responses) if x.startswith("答案") or x.startswith("answer")]
        predictions = []
        for resp_quest, resp_ans in zip(responses_questions, responses_answers):
            if return_form == "list":
                pred = response_2_prediction_of_list(args, query, resp_ans, question=resp_quest)
            elif return_form in ["dict", "json"]:
                pred = response_2_prediction_of_dict_json(args, query, resp_ans, question=resp_quest, return_form=return_form)
            else:
                raise ValueError(f"Unrecognized return_form={return_form}")
            predictions.append(pred)
    
    if return_responded_qa:
        return (predictions, responses_questions, responses_answers)
    else:
        return predictions


def response_2_prediction(args, query, response, resp_idx=None, question=None, return_form="dict", complete_form="question", return_responded_qa=False):
    if complete_form == "question": # 补全每个问题的答案
        if return_form in ["dict", "json"]:
            prediction = response_2_prediction_of_dict_json(args, query, response, resp_idx=resp_idx, question=question, return_form=return_form)
        elif return_form == "list":
            prediction = response_2_prediction_of_list(args, query, response, resp_idx=resp_idx, question=question)
        else:
            raise ValueError(f"Unrecognized return_form: {return_form}")
        return prediction
    elif complete_form == "dialogue": # 补全整个对话，包括所有问题和答案
        predictions = response_of_dialogue_2_prediction(args, query, response, return_form=return_form, return_responded_qa=return_responded_qa)
        return predictions
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

def majority_voting(args, prediction_ls=None, cnt_prediction_tuple=None):
    '''
    Vote for most consistent named entities from a set of predictions.
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
    
    if cnt_prediction_tuple is None:
        assert prediction_ls is not None
        # 对所有(mention, type)对计数
        cnt_prediction_tuple = {}
        for prediction in prediction_ls:
            for k, v in prediction.items():
                if (k,v) not in cnt_prediction_tuple:
                    cnt_prediction_tuple[(k,v)] = 0
                cnt_prediction_tuple[(k,v)] += 1
    # 将票数超过n/2的答案加入最终预测结果
    prediction_voted = {}
    for (k,v) in cnt_prediction_tuple:
        if cnt_prediction_tuple[(k,v)] >= lowest_votes_for_O:
            prediction_voted[k] = v

    return prediction_voted

def combine_question_predictions(args, predictions, return_form="dict"):
    prediction = {}
    if return_form in ["dict"]:
        for tmp_preds in predictions:
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