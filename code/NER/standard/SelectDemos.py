import os
import json
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import traceback
from openai.embeddings_utils import cosine_similarity
import tiktoken
import argparse
from sklearn.cluster import KMeans


def load_train_data(path):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def euclideanDist(a, b):
    return np.sqrt(sum((a - b) ** 2))

def select_demos_by_cluster_kmeans(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb):
    demo_size = args.demo_size
    if args.demo_select_method == "GPTEmbClusterKmeans":
        train_embs = train_GPTEmb
    elif args.demo_select_method == "SBERTEmbClusterKmeans":
        train_embs = train_SBERTEmb

    kmeans = KMeans(n_clusters=demo_size, init="k-means++", random_state=42)

    print("fitting......")
    print("Start time: {}".format(time.strftime("%m%d%H%M")))
    kmeans.fit(train_embs)
    end_time = time.strftime("%m%d%H%M")
    print("End time: {}".format(time.strftime("%m%d%H%M")))

    labels = kmeans.labels_

    # 获取每个cluster的mean向量
    label_set = sorted(list(np.unique(labels)))
    means = np.zeros((len(label_set), train_embs.shape[1]))
    for i, lab in enumerate(label_set):
        means[i] = np.mean(train_embs[labels==lab], axis=0)

    # 计算每个样本与其cluster mean的距离
    dists = []
    for i, emb in enumerate(tqdm(train_embs, desc="compute dist to cluster mean.")):
        lab = labels[i]
        dist = euclideanDist(emb, means[label_set.index(lab)])
        dists.append(dist)

    label2closest_dist = dict(zip(label_set, [float("inf")]*len(label_set)))
    label2closest_idx = dict(zip(label_set, [-1]*len(label_set)))
    # 找到每个cluster中距离cluster mean最近的样本
    for i, dist in enumerate(tqdm(dists, desc="find closest sample to cluster mean.")):
        lab = labels[i]
        if label2closest_dist[lab] > dist:
            label2closest_dist[lab] = dist
            label2closest_idx[lab] = i

    demos_selected = []
    demos_parse_selected = []
    demos_selected_GPTEmb = []
    demos_selected_SBERTEmb = []
    for _, idx in label2closest_idx.items():
        demos_selected.append(train_data[idx])
        demos_parse_selected.append(train_data_parse[idx])
        demos_selected_GPTEmb.append(train_embs[idx])
        demos_selected_SBERTEmb.append(train_SBERTEmb[idx])

    demos_selected_GPTEmb = np.stack(demos_selected_GPTEmb, axis=0)
    demos_selected_SBERTEmb = np.stack(demos_selected_SBERTEmb, axis=0)

    return demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb


def select_demos_by_cluster_votek(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb):
    pass

def select_demos_by_random(demo_size, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb):
    demos_selected = []
    demos_parse_selected = []
    demos_idx = []
    demos_selected_GPTEmb = []
    demos_selected_SBERTEmb = []
    while len(demos_selected) < demo_size:
        tmp_idx = random.choice(range(len(train_data)))
        # 样例不重复
        if tmp_idx in demos_idx:
            continue

        # 不选不含实体标签的样本
        if len(train_data[tmp_idx]["label"]) == 0:
            continue

        demos_selected.append(train_data[tmp_idx])
        demos_parse_selected.append(train_data_parse[tmp_idx])
        demos_idx.append(tmp_idx)
        demos_selected_GPTEmb.append(train_GPTEmb[tmp_idx])
        demos_selected_SBERTEmb.append(train_SBERTEmb[tmp_idx])

    demos_selected_GPTEmb = np.stack(demos_selected_GPTEmb, axis=0)
    demos_selected_SBERTEmb = np.stack(demos_selected_SBERTEmb, axis=0)

    return demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb

def select_demos(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb):
    if "EmbClusterKmeans" in args.demo_select_method:
        demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb = select_demos_by_cluster_kmeans(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb)
    if "GPTEmbClusterVoteK" in args.demo_select_method:
        demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb = select_demos_by_cluster_votek(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb)
    elif args.demo_select_method == "random":
        demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb = select_demos_by_random(args.demo_size, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb)
    
    return demos_selected, demos_parse_selected, demos_selected_GPTEmb, demos_selected_SBERTEmb

def save_demos(path, demos):
    if path.endswith(".txt") or path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(demos, indent=4, ensure_ascii=False))
    else:
        raise ValueError(f"Wrong path for prompts saving: {path}")
    
def save_embs(path, embs):
    np.save(path, embs)


def main(args):
    # 设定随机数种子
    random.seed(args.random_seed)

    # 加载数据
    train_data = load_train_data(args.train_data_path)
    train_data_parse = load_train_data(args.train_data_parse_path)
    train_GPTEmb = np.load(args.train_GPTEmb_path)
    train_SBERTEmb = np.load(args.train_SBERTEmb_path)

    # 加载embedding
    # if args.few_shot_setting in ["pool", "fixed"]:
    #     train_embs = np.load(args.train_embs_path)
    #     assert len(train_data) == len(train_embs)
    # else:
    #     train_embs = None
    
    # 挑选样例
    demo_data, demo_data_parse, demos_selected_GPTEmb, demos_selected_SBERTEmb = select_demos(args, train_data, train_data_parse, train_GPTEmb, train_SBERTEmb)

    # 存储demo、配套parse的demo、demo emb
    save_demos(args.demo_data_path, demo_data)
    save_demos(args.demo_data_parse_path, demo_data_parse)
    save_embs(args.demo_GPTEmb_path, demos_selected_GPTEmb)
    save_embs(args.demo_SBERTEmb_path, demos_selected_SBERTEmb)


def get_paths(args):
    # 选择embedding
    # if "GPTEmb" in args.demo_select_method:
    #     emb = "GPTEmb"
    # elif "SBERT" in args.demo_select_method:
    #     emb = "SBERT"
    # else:
    #     emb = None

    # query数据路径
    # 原始数据
    args.train_data_path = f"OPENAI/data/{args.task}/{args.dataname}/train.json"
    # 配套parse的数据
    args.train_data_parse_path = f"OPENAI/data/{args.task}/{args.dataname}/train_parse_hanlp.json"
    # 数据emb
    args.train_GPTEmb_path = f"OPENAI/data/{args.task}/{args.dataname}/train_GPTEmb.npy" 
    args.train_SBERTEmb_path = f"OPENAI/data/{args.task}/{args.dataname}/train_SBERTEmb.npy" 
    
    # demo存储路径
    demo_select_method = args.demo_select_method
    if demo_select_method=="random":
        demo_select_method = f"{args.demo_select_method}_{args.random_seed}"
    if args.few_shot_setting in ["fixed", "pool"]:
        demo_folder = f"demo_{args.few_shot_setting}"
        demo_dir = f"OPENAI/data/{args.task}/{args.dataname}/{demo_folder}"
        if not os.path.exists(demo_dir):
            os.makedirs(demo_dir)
        # 原始demo
        demo_filename = f"train_demo_{args.few_shot_setting}_{demo_select_method}_{args.demo_size}.json"
        args.demo_data_path = f"{demo_dir}/{demo_filename}"
        # 配套parse的demo
        demo_parse_filename = f"train_demo_{args.few_shot_setting}_{demo_select_method}_{args.demo_size}_parse_hanlp.json"
        args.demo_data_parse_path = f"{demo_dir}/{demo_parse_filename}"
        # demo emb
        demo_GPTEmb_filename = f"train_demo_{args.few_shot_setting}_{demo_select_method}_{args.demo_size}_GPTEmb.npy"
        args.demo_GPTEmb_path = f"{demo_dir}/{demo_GPTEmb_filename}"
        demo_SBERTEmb_filename = f"train_demo_{args.few_shot_setting}_{demo_select_method}_{args.demo_size}_SBERT.npy"
        args.demo_SBERTEmb_path = f"{demo_dir}/{demo_SBERTEmb_filename}"
    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--task", default="NER")
    # retrieval
    parser.add_argument("--few_shot_setting", default="fixed", choices=["fixed", "pool"])
    parser.add_argument("--demo_size", default=10, type=int)
    parser.add_argument("--demo_select_method", default="random", choices=["random", "GPTEmbClusterKmeans", "GPTEmbClusterVoteK"])
    parser.add_argument("--random_seed", default=137, type=int)

    args = parser.parse_args()


    args = get_paths(args)

    print("---------- Select demos ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

