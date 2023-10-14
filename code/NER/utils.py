import logging, logging.config
import sys
import os
import json
import torch
from tqdm import tqdm
import pandas as pd

def copy_file_to_path(data, dir, path):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.exists(path):
        raise ValueError(f"Path already exists: {path}")
    save_data(path, data)

def load_data(path):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def save_data(path, data):
    if path.endswith(".txt"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data[:-1]:
                f.write(str(item) + "\n")
            f.write(str(data[-1]))
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        raise ValueError(f"Wrong path for prompts saving: {path}")
    
def json2dict(json_list):
    d = dict()
    for item in json_list:
        k = list(item.keys())[0]
        v = item[k]
        d[k] = v
    
    return d

def dict2json(in_dict):
    out_json = []
    for k, v in in_dict.items():
        out_json.append(
            {k: v}
        )
    return out_json

def format_json2str(in_json):
    out_str = "["
    for i_item, item in enumerate(in_json):
        k = list(item.keys())[0]
        v = item[k]
        out_str += "{\"%s\": \"%s\"}" % (k, v)        
        if i_item < len(in_json)-1:
            out_str += ", "
    out_str += "]"

    return out_str


def fetch_indexed_embs(train_embs, demo_data):
    '''
    train_embs: embs of whole training set.
    demo_data: demo set selected from training set.
    '''
    idxes = [x["idx"] for x in demo_data]
    idxes = torch.tensor(idxes).long()
    train_embs = torch.from_numpy(train_embs)
    demo_embs = torch.index_select(train_embs, dim=0, index=idxes)
    demo_embs = demo_embs.numpy()

    return demo_embs

def get_logger(file_name, log_dir, config_dir):
    config_dict = json.load(open( os.path.join(config_dir, 'log_config.json')))
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir,  file_name.replace('/', '-'))
    # config_dict['handlers']['file_handler']['level'] = "INFO"
    config_dict['root']['level'] = "INFO"
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger()

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


from typing import List, Dict

# 分别统计gold和pred的情况，包括cound, rate, p/r/f1
def compute_metrics(args, data_responses:List[Dict], write_metric:bool=True):
    '''
    columns to be collected:
        ["Type", "Gold count", "Gold rate", "Pred count", "Pred rate", "Prec", "Rec", "F1"]
    '''
    
    id2label = args.id2label
    # 每个类别一个记录，一条记录包含所有要收集的字段
    type2record = {}
    for lb in id2label:
        type2record[lb] = {"Type":lb, "Gold count":0, "Gold rate":0, "Pred count":0, "Pred rate":0, "Correct count":0, "Prec":0, "Rec":0, "F1":0}
    
    for i_item, item in enumerate(tqdm(data_responses, desc="compute metric")):
        # 计算pool size以内cconfident annotation的准确率
        if args.confident and args.confident_sample_size:
            if i_item >= args.confident_sample_size:
                break

        curr_label = item["label"]
        if isinstance(curr_label, str):
            curr_label = eval(curr_label)
        if isinstance(curr_label, list):
            curr_label = json2dict(curr_label)
        curr_pred = item["prediction"]
        if isinstance(curr_pred, str):
            curr_pred = eval(curr_pred)
        if isinstance(curr_pred, list):
            curr_pred = json2dict(curr_pred)

        # 去掉 空字符“” 的预测
        if "" in curr_pred:
            del curr_pred[""]
        
        for tmp_mention, tmp_type in curr_label.items():
            type2record[tmp_type]["Gold count"] += 1
        
        ood_type_preds = []
        ood_mention_preds = []
        for tmp_mention, tmp_type in curr_pred.items():
            # ood type
            if tmp_type not in id2label:
                ood_type_preds.append({tmp_mention:tmp_type})
                continue
            type2record[tmp_type]["Pred count"] +=1
            # ood mention
            if tmp_mention not in item["sentence"]:
                ood_mention_preds.append({tmp_mention:tmp_type})
                continue
            # 与gold label对比
            if tmp_mention in curr_label and tmp_type == curr_label[tmp_mention]:
                type2record[tmp_type]["Correct count"] += 1

        # 打印ood
        # if len(ood_type_preds)>0:
        #     logger.info(f"OOD Type predictions:\n{ood_type_preds}")
        # if len(ood_mention_preds)>0:
        #     logger.info(f"OOD mention predictions:\n{ood_mention_preds}")
    
    # 计算总体指标
    n_gold_tot = sum([x["Gold count"] for x in type2record.values()])
    n_pred_tot = sum([x["Pred count"] for x in type2record.values()])
    n_correct_tot = sum([x["Correct count"] for x in type2record.values()])
    prec_tot = n_correct_tot / n_pred_tot if n_pred_tot else 0
    rec_tot = n_correct_tot / n_gold_tot if n_gold_tot else 0
    if prec_tot and rec_tot:
        f1_tot = 2*prec_tot*rec_tot / (prec_tot+rec_tot)
    else:
        f1_tot = 0
    prec_tot = round(prec_tot,4)*100
    rec_tot = round(rec_tot,4)*100
    f1_tot = round(f1_tot,4)*100

    # 计算每一类的指标
    for k in type2record:
        gold_count = type2record[k]["Gold count"]
        pred_count = type2record[k]["Pred count"]
        correct_count = type2record[k]["Correct count"]
        
        gold_rate = gold_count / n_gold_tot if n_gold_tot else 0
        pred_rate = pred_count / n_pred_tot if n_pred_tot else 0
        gold_rate = round(gold_rate,4)*100
        pred_rate = round(pred_rate,4)*100

        prec = correct_count / pred_count if pred_count else 0
        rec = correct_count / gold_count if gold_count else 0
        if prec and rec:
            f1 = 2*prec*rec / (prec+rec)
        else:
            f1 = 0
        prec = round(prec,4)*100
        rec = round(rec,4)*100
        f1 = round(f1,4)*100

        type2record[k]["Gold rate"] = gold_rate
        type2record[k]["Pred rate"] = pred_rate
        type2record[k]["Prec"] = prec
        type2record[k]["Rec"] = rec
        type2record[k]["F1"] = f1

    type2record["Total"] = {"Type":"ToTal", "Gold count":n_gold_tot, "Gold rate":100, "Pred count":n_pred_tot, "Pred rate":100, "Correct count":n_correct_tot, "Prec":prec_tot, "Rec":rec_tot, "F1":f1_tot}

    # 转为表格形式
    df_metrics = pd.DataFrame(list(type2record.values()))
    # 打印指标
    logger.info(f"===== Metrics =====\n{df_metrics}")
    # 将指标写入文件
    if write_metric:
        metric_path = args.metric_path
        df_metrics.to_csv(metric_path, index=False)

# 加载已经计算好的类别信息，包括每一类的count, rate, len_mean, etc.
def compute_metrics_v0(args, data_responses:List[Dict], write_metric:bool=True):
    class_n_r = pd.read_csv(args.class_n_r_path)
    mode = args.datamode
    if args.dataname == "conll2003":
        mode = "conllpp_test"
    if "train_shuffle" in mode:
        mode = "train"
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
    for i_item, item in enumerate(tqdm(data_responses, desc="compute mmetric")):
        # 计算pool size以内cconfident annotation的准确率
        if args.confident and args.confident_sample_size:
            if i_item >= args.confident_sample_size:
                break
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
    print("OOD Types:")
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
    if write_metric:
        metric_path = args.metric_path
        metrics_table.to_csv(metric_path, index=False)

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
    # if write_metric:
        # twostage_path = args.twostage_path
        # metrics_table_twostage.to_csv(twostage_path, index=False)

import tiktoken
import logging
logger = logging.getLogger()
from const import my_openai_api_keys, model_list

def set_api_key(model_name, ports=None):
    model_publisher = model_list[model_name]["publisher"]
    if  model_publisher == "openai":
        api_keys = my_openai_api_keys
    else:
        assert ports != None
        api_keys = []
        for port in ports:
            api_keys.append({
                "key":"empty", 
                "set_base":True, 
                "api_base":"http://localhost:%s/v1" % port
            })
    
    return api_keys


def assert_gpt35_turbo_16k(args, chat_paradigm="standard"):
    if chat_paradigm == "standard":
        if args.few_shot_setting != "zs":
            if args.reason_hint is None and args.tool_aug is None:
                if args.few_shot_number >= 35:
                    assert args.model == "gpt-3.5-turbo-16k"
            else:
                if args.few_shot_number >= 10:
                    assert args.model == "gpt-3.5-turbo-16k"
    elif chat_paradigm == "qa_dialogue":
        if args.demo_form == "entity":
            if args.few_shot_setting != "zs":
                if args.reason_hint is None and args.tool_aug is None:
                    if args.few_shot_number >= 35:
                        assert args.model == "gpt-3.5-turbo-16k"
                else:
                    if args.few_shot_number >= 16:
                        assert args.model == "gpt-3.5-turbo-16k"
        elif args.demo_form == "dialogue":
            if args.few_shot_setting != "zs":
                if args.reason_hint is None and args.tool_aug is None:
                    if args.few_shot_number >= 5:
                        assert args.model == "gpt-3.5-turbo-16k"
                else:
                    if args.few_shot_number >= 3:
                        assert args.model == "gpt-3.5-turbo-16k"
        elif args.demo_form == "standard":
            if args.few_shot_setting != "zs":
                if args.reason_hint is None and args.tool_aug is None:
                    if args.few_shot_number >= 16:
                        assert args.model == "gpt-3.5-turbo-16k"
                else:
                    if args.few_shot_number >= 8:
                        assert args.model == "gpt-3.5-turbo-16k"
        else:
            raise ValueError(f"Unrecognized demo_form: {args.demo_form}")
    elif chat_paradigm == "qa_divide_combine":
        if args.few_shot_setting != "zs":
            if args.reason_hint is None and args.tool_aug is None:
                if args.few_shot_number >= 20:
                    assert args.model == "gpt-3.5-turbo-16k"
            else:
                if args.few_shot_number >= 10:
                    assert args.model == "gpt-3.5-turbo-16k"
    elif chat_paradigm == "qa_dialogue_recur_input":
        raise "Unimplemented part!"
    elif chat_paradigm == "self_verification":
        # 暂时SV用不上turbo-16k
        # assert args.model == "gpt-3.5-turbo"
        pass
    else:
        raise ValueError(f"Unrecognized chat_paradigm:{chat_paradigm}")


def max_tokens(model):
    n_max_tokens = model_list[model]["max_tokens"]
    return n_max_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        print("Warning: Not an official model of OpenAI. Set tokens_per_message, tokens_per_name = 0, 0")
        tokens_per_message, tokens_per_name = 0, 0

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


import time
import random
import openai
import logging
logger = logging.getLogger()

def run_llm(
        messages, 
        openai_key=None, 
        model_name="gpt-3.5-turbo", 
        temperature=0,
        stop=None,
        ports=None
):
    agent = SafeOpenai(openai_key, model_name=model_name,ports=ports)
    if "text" in model_name:
        # 将messages格式转换为prompt格式
        content_all = [x["content"] for x in messages]
        prompt = "\n".join(content_all)
        response = agent.text(
            model=model_name, 
            prompt=prompt, 
            temperature=temperature, 
            stop=stop,
            return_text=True
            )
    else:
        response = agent.chat(
            model=model_name, 
            messages=messages, 
            temperature=temperature, 
            stop=stop,
            return_text=True
            )
    return response

class SafeOpenai:
    def __init__(self, keys=None, model_name=None, start_id=None, proxy=None, ports=None):
        # if model_list[model_name]["publisher"] == "openai":
        #     self.llm_server = "openai"
        # else:
        #     self.llm_server = "local"

        if keys is None:
            raise "Please provide OpenAI Key."
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        if start_id is None:
            start_id = random.sample(list(range(len(keys))), 1)[0]
            # # TODO: debug
            # start_id = 0
            
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.keys)
        current_key = self.keys[self.key_id % len(self.keys)]
        openai.api_key = current_key["key"]
        if "org" in current_key:
            openai.organization = current_key["org"]
        if "set_base" in current_key and current_key["set_base"] is True:
            openai.api_base = current_key["api_base"]

        # # 如果模型不来自openai
        # self.ports = ports
        # if self.llm_server == "local":
        #     assert ports != None
        #     # 本地可以同时部署多个llm,各自使用不同的port
        #     self.port_id = random.sample(list(range(len(ports))),1)[0]
        #     openai.api_base = "http://localhost:%s/v1" % ports[self.port_id]
    
    def set_next_api_key(self):
        self.key_id = (self.key_id + 1) % len(self.keys)
        current_key = self.keys[self.key_id]
        openai.api_key = current_key["key"]
        if "org" in current_key:
            openai.organization = current_key["org"]
        if "set_base" in current_key and current_key["set_base"] is True:
            openai.api_base = current_key["api_base"]
        
    def chat(self, 
            model, 
            messages, 
            temperature, 
            stop=None,
            return_text=False, 
            sleep_seconds=2, 
            max_read_timeout=1, 
            max_http=1):
        timeout_cnt = 0
        http_cnt = 0
        while True:
            if timeout_cnt >= max_read_timeout:
                logger.info(f"timeout_cnt exceed max_read_timeout {max_read_timeout}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
            elif http_cnt >= max_http:
                logger.info(f"http_cnt exceed max_read_timeout {max_http}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."                            
            try:
                completion = openai.ChatCompletion.create(
                    model=model, 
                    messages=messages, 
                    temperature=temperature,
                    stop=stop,
                    timeout=3)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                if "maximum context length" in str(e): # 若input长度超出限制，则直接return
                    logger.info(f"###### Exceed length, set response to empty str.")
                    return ""
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.APIConnectionError as e:
                logger.info(f"Failed to connect to OpenAI API: {e}")
                if "HTTPSConnectionPool" in str(e) and 'Remote end closed connection without response' in str(e):
                    http_cnt += 1
                    logger.info(f"http_cnt + 1 = {http_cnt}")                
                self.set_next_api_key()
                time.sleep(sleep_seconds)  
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}\n{self.keys[self.key_id]}")
                if "please check your plan and billing details" in str(e):
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1
                self.set_next_api_key()
                time.sleep(sleep_seconds)  
            except Exception as e:
                logger.info(str(e))
                # 删去被封的keys
                if "This key is associated with a deactivated account" in str(e):
                    print(f"deactivated error key: {self.keys[self.key_id]}")
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1

                    if self.key_id + 1 == len(self.keys):
                        logger.info("All keys are invalid!!!")
                        sys.exit()
                                                           
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "Read timed out" in str(e):
                    timeout_cnt += 1
                    logger.info(f"timeout_cnt + 1 = {timeout_cnt}")

                # if self.llm_server == "local":
                #     # 本地可以同时部署多个llm,各自使用不同的port
                #     self.port_id = (self.port_id+1) % len(self.ports)
                #     openai.api_base = "http://localhost:%s/v1" % self.ports[self.port_id]
                # else:
                #     raise ValueError(f"Unrecognized llm_server: {self.llm_server}")

                self.set_next_api_key()
                time.sleep(sleep_seconds) # TODO: 语句执行顺序？

        if return_text:
            completion = completion['choices'][0]['message']['content']
        return completion

    def text(self, 
            model, 
            prompt, 
            temperature, 
            stop=None,
            return_text=False, 
            sleep_seconds=2, 
            max_read_timeout=1, 
            max_http=1):
        timeout_cnt = 0
        http_cnt = 0
        while True:
            if timeout_cnt >= max_read_timeout:
                logger.info(f"timeout_cnt exceed max_read_timeout {max_read_timeout}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
            elif http_cnt >= max_http:
                logger.info(f"http_cnt exceed max_read_timeout {max_http}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."                            
            try:

                completion = openai.Completion.create(
                    model=model, 
                    prompt=prompt, 
                    temperature=temperature,
                    stop=stop,
                    timeout=3)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                if "maximum context length" in str(e): # 若input长度超出限制，则直接return
                    logger.info(f"###### Exceed length, set response to empty str.")
                    return ""
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.APIConnectionError as e:
                logger.info(f"Failed to connect to OpenAI API: {e}")
                if "HTTPSConnectionPool" in str(e) and 'Remote end closed connection without response' in str(e):
                    http_cnt += 1
                    logger.info(f"http_cnt + 1 = {http_cnt}")
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}\n{self.keys[self.key_id]}")
                if "please check your plan and billing details" in str(e):
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except Exception as e:
                logger.info(str(e))
                if "This key is associated with a deactivated account" in str(e):
                    print(f"deactivated error key: {self.keys[self.key_id]}")
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1

                    if self.key_id + 1 == len(self.keys):
                        logger.info("All keys are invalid!!!")
                        sys.exit()
                                                           
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "Read timed out" in str(e):
                    timeout_cnt += 1
                    logger.info(f"timeout_cnt + 1 = {timeout_cnt}")

                self.set_next_api_key()
                time.sleep(sleep_seconds)
                
        if return_text:
            completion = completion['choices'][0]['text']
        return completion

    def text_v0(self, *args, return_text=False, sleep_seconds=2, max_timeout=3, **kwargs):
        timeout_cnt = 0
        while True:
            if timeout_cnt >= max_timeout:
                logger.info(f"timeout_cnt exceed max_timeout {max_timeout}. Skip this request.")
                return f"Timeout exceeds {max_timeout}, skip."            
            try:
                completion = openai.Completion.create(*args, **kwargs)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                self.key_id = (self.key_id + 1) % len(self.keys)
                openai.organization = self.keys[self.key_id]["org"]
                openai.api_key = self.keys[self.key_id]["key"]
                time.sleep(sleep_seconds)                
            except openai.error.APIConnectionError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                self.key_id = (self.key_id + 1) % len(self.keys)
                openai.organization = self.keys[self.key_id]["org"]
                openai.api_key = self.keys[self.key_id]["key"]
                time.sleep(sleep_seconds)  
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                self.key_id = (self.key_id + 1) % len(self.keys)
                openai.organization = self.keys[self.key_id]["org"]
                openai.api_key = self.keys[self.key_id]["key"]
                time.sleep(sleep_seconds)  
            except Exception as e:
                logger.info(str(e))
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "" in str(e):
                    logger.info("timeout_cnt + 1")
                    timeout_cnt += 1
                
                self.key_id = (self.key_id + 1) % len(self.keys)
                openai.organization = self.keys[self.key_id]["org"]
                openai.api_key = self.keys[self.key_id]["key"]
                time.sleep(sleep_seconds)
                
        if return_text:
            completion = completion['choices'][0]['text']
        return completion
