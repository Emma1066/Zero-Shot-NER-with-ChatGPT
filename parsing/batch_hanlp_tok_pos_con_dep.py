import hanlp
import json
from tqdm import tqdm


def get_parse_per_sentence(sample, parser):
    doc = parser(
        sample["sentence"],
        tasks=["tok*", "pos", "con", "dep"]
    )

    tok = doc["tok/fine"]
    pos = doc["pos/ctb"]
    dep = doc["dep"]
    con = doc["con"]
    tok_coarse = doc["tok/coarse"]

    # arrange POS output format
    tok_pos_pair = [[t,p] for t,p in zip(tok, pos)]
    tok_pos_pair = [f"{t}/{p}" for [t,p] in tok_pos_pair]
    tok_pos_pair_str = " ".join(tok_pos_pair)
    # arrange dependency tree output format
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
DATANAME = "msra_5_samples"
MODE = "test"

datafolder = DATANAME
datamode = MODE
indata_path = f"data/{datafolder}/{datamode}.json"
outdata_path = f"data/{datafolder}/{datamode}_parse_hanlp.json"

indata = json.load(open(indata_path, "r", encoding="utf-8"))

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

print("========== HANLP parsing ===========")
print(f"save path: {outdata_path}")
data_parse = []
for i_item, item in enumerate(tqdm(indata, desc="get parse")):
    item_parse = get_parse_per_sentence(item, HanLP)
    data_parse.append(item_parse)

with open(outdata_path, "w", encoding="utf-8") as wf:
    wf.write(json.dumps(data_parse, indent=4, ensure_ascii=False))

print(f"file saved to: {outdata_path}")