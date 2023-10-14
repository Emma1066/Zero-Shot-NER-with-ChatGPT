import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse
import openai
from openai.embeddings_utils import get_embedding
from sentence_transformers import SentenceTransformer

def load_query_data(path):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def load_demo_data(path, demo_num):
    if demo_num:
        demo_data = json.load(open(path, "r", encoding="utf-8"))
    else:
        demo_data = list()

    return demo_data


def generate_emb_gpt(args, emb_model, query_data):
    text_to_encode = [x["sentence"] for x in query_data]
    num = len(text_to_encode)
    embeddings = []
    bar = tqdm(range(0,num,args.batch_size),desc='calculate embeddings')
    for i in range(0,num,args.batch_size):
        embeddings += emb_model.encode(text_to_encode[i:i+args.batch_size]).tolist()
        bar.update(1)
    embeddings = np.array(embeddings)
    mean_embeddings = np.mean(embeddings, 0, keepdims=True)
    embeddings = embeddings - mean_embeddings
    return embeddings



def save_embs(path, embs):
    np.save(path, embs)

def main(args):
    # 加载数据
    query_data = load_query_data(args.query_data_path)

    # 加载模型
    model = SentenceTransformer(args.emb_model)

    # 生成embedding
    embs = generate_emb_gpt(args, model, query_data)
    print(f"ems shape: {embs.shape}")
    save_embs(args.query_embs_path, embs)


def get_paths(args):
    # query数据路径
    args.query_data_path = f"OPENAI/data/{args.task}/{args.dataname}/{args.datamode}.json"
    # embedding路径
    args.query_embs_path = f"OPENAI/data/{args.task}/{args.dataname}/{args.datamode}_SBertEmb.npy"
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--datamode", default=None, type=str, choices=["train", "test"])
    parser.add_argument("--task", default="NER")
    # 模型
    parser.add_argument("--emb_model", default="sentence-transformers/all-mpnet-base-v2", type=str)
    parser.add_argument("--batch_size", default=16, type=int)

    args = parser.parse_args()

    args = get_paths(args)

    print("---------- Generate SBert emb ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

