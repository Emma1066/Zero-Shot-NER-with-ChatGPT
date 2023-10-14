import hanlp
import json
from tqdm import tqdm


def get_parse_per_sentence(sample, parser):
    doc = parser(
        sample["sentence"],
        tasks=["tok*", "pos", "con", "dep"]
    )

    # 自定义输出形式
    tok = doc["tok/fine"]
    pos = doc["pos/ctb"]
    dep = doc["dep"]
    con = doc["con"]
    tok_coarse = doc["tok/coarse"]

    # 输出POS的可读形式
    tok_pos_pair = [[t,p] for t,p in zip(tok, pos)]
    tok_pos_pair = [f"{t}/{p}" for [t,p] in tok_pos_pair]
    tok_pos_pair_str = " ".join(tok_pos_pair)
    # 输出DEP的可读形式
    trip_dep = []
    for i_item, item in enumerate(dep):
        '''[head_idx, dep_rel]'''
        tmp_tok = tok[i_item]
        tmp_head_idx = item[0]-1
        tmp_dep_rel = item[1]
        tmp_head = tok[tmp_head_idx]
        trip_dep.append(
            [tmp_tok, tmp_head, tmp_dep_rel]
        )

    # parse_dict = {
    #     "tok/fine": tok,
    #     "pos/ctb": pos,
    #     "dep": dep,
    #     "con": con,
    #     "tok_pos_pair": tok_pos_pair,
    #     "trip_dep": trip_dep,
    #     "con_str": con_str
    # }

    sample["tok/fine"] = str(tok)
    sample["tok/coarse"] = str(tok_coarse)
    sample["pos/ctb"] = str(pos)
    sample["dep"] = str(dep)
    sample["tok_pos_pair"] = str(tok_pos_pair)
    sample["tok_pos_pair_str"] = tok_pos_pair_str
    sample["trip_dep"] = str(trip_dep)
    sample["con_str"] = str(con)

    return sample


# load data
DATANAME = "wnut17"
folder = 0
datafolder = DATANAME
if DATANAME == "ace04en":
    datafolder = f"{DATANAME}/{folder}"
MODE = "train"

datamode = MODE
# if DATANAME == "conll2003":
#     datamode = "conllpp_test"
indata_path = f"OPENAI/data/NER/{datafolder}/{datamode}.json"
outdata_path = f"OPENAI/data/NER/{datafolder}/{datamode}_parse_hanlp.json"

indata = json.load(open(indata_path, "r", encoding="utf-8"))

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

# with open(outdata_path,"wb+", buffering=0) as realtime_f:
#     for i_item, item in enumerate(indata):
#         parse_dict = get_parse_per_sentence(item["sentence"], HanLP)
#         item["tok/fine"] = str(parse_dict["tok/fine"])
#         item["pos/ctb"] = str(parse_dict["pos/ctb"])
#         item["dep"] = str(parse_dict["dep"])
#         item["con"] = str(parse_dict["con"])

#         realtime_f.write((str(item)+"\n").encode("utf-8"))
print("========== HANLP parsing ===========")
print(f"save path: {outdata_path}")
data_parse = []
for i_item, item in enumerate(tqdm(indata, desc="get parse")):
    # debug
    # if i_item > 3:
    #     break
    item_parse = get_parse_per_sentence(item, HanLP)
    data_parse.append(item_parse)

with open(outdata_path, "w", encoding="utf-8") as wf:
    wf.write(json.dumps(data_parse, indent=4, ensure_ascii=False))

print(f"file saved to: {outdata_path}")