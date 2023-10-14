# model_abb_map = {
#     "gpt-3.5-turbo": "tb",
#     "gpt-3.5-turbo-16k": "tb",
#     "gpt-3.5-turbo-0613": "tb0613",
#     "text-davinci-003": "d3",
#     "llama-2-13b-chat-hf": "lm2_13c",
#     "llama-2-70b-chat-hf": "lm2_70c",
#     "Llama-2-70B-chat-GPTQ": "lm2_70c_gptq",
# }
model_list = {
    "gpt-3.5-turbo": {"abbr":"tb", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-16k": {"abbr":"tb", "publisher":"openai", "max_tokens":4096*4},
    "gpt-3.5-turbo-0613": {"abbr":"tb0613", "publisher":"openai", "max_tokens":4096},
    "text-davinci-003": {"abbr":"d3", "publisher":"openai", "max_tokens":4097},
    "llama-2-13b-chat-hf": {"abbr":"lm2_13c", "publisher":"meta", "max_tokens":4096},
    "llama-2-70b-chat-hf": {"abbr":"lm2_70c", "publisher":"meta", "max_tokens":4096},
    "Llama-2-70B-chat-GPTQ": {"abbr":"lm2_70c_gptq", "publisher":"meta", "max_tokens":4096},
}

dataset_language_map = {
    "bc5cdr": "en",
    "bc5cdr_300_42": "en",
    "bc5cdr_300_52": "en",
    "bc5cdr_300_137": "en",
    "JNLPBA_300_42": "en",
    "JNLPBA_300_52": "en",
    "JNLPBA_300_137": "en",
    "NCBI_100_42": "en",
    "conll2003": "en",
    "conll2003_300_42": "en",
    "conll2003_300_52": "en",
    "conll2003_300_137": "en",
    "conll2003pp_300_42": "en",
    "conll2003pp_300_52": "en",
    "conll2003pp_300_137": "en",
    "wnut17_300_42": "en",
    "ontonotes5en_300_42": "en",
    "genia_300_42": "en",
    "genia": "en",
    "scierc": "en",
    "ace05en": "en",
    "ace05en_300_42": "en",
    "ace05en_300_52": "en",
    "ace05en_300_137": "en",
    "ace04en": "en",
    "ace04en_0_300_42": "en",
    "ontonotes5en": "en",
    "msra": "zh",
    "msra_300_42": "zh",
    "weibo": "zh",
    "PowerPlantFlat": "zh",
    "PowerPlantNested": "zh",
    "resume": "zh",
    "ontonotes4zh": "zh",
    "ontonotes4zh_300_42": "zh",
    "ontonotes4zh_300_52": "zh",
    "ontonotes4zh_300_137": "zh",
    "FEWNERD_300_42": "en",
    "wikigold": "en",
    "bionlp11id_300_42": "en",
    "bionlp11id_300_52": "en",
    "bionlp11id_300_137": "en",
    "bionlp13pc_300_42": "en",
    "craft_300_42": "en",
    "craft_300_52": "en",
    "craft_300_137": "en",
    "wnut17": "en"
}

dataset_label_order_map = {
    "PowerPlantFlat": {
        "0": [["设备名称", "设备标识"],["系统名称", "系统标识"],["部件名称", "地点", "人员"]],
        "1": [["设备名称"], ["设备标识"],["系统名称"], ["系统标识"],["部件名称"], ["地点"], ["人员"]],
        "2": [["设备标识"],["设备名称"], ["系统标识"], ["系统名称"], ["部件名称"], ["地点"], ["人员"]],
        "3": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称"], ["地点"], ["人员"]],
        "4": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称", "地点", "人员"]],
        "5": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称"], ["地点", "人员"]],
        "chatgpt0": [["地点"],["系统名称"], ["系统标识"], ["设备名称"], ["设备标识"], ["部件名称"], ["人员"]],
        "chatgpt1": [["地点"],["系统名称", "系统标识"], ["设备名称", "设备标识"], ["部件名称"], ["人员"]]
    },
    "PowerPlantNested":{
        '2': [["设备标识"],["设备名称"], ["系统标识"], ["系统名称"], ["部件名称"], ["地点"], ["人员"],["反应堆状态"],["电站事件"]],        
        "3": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称"], ["地点"], ["人员"],["反应堆状态"],["电站事件"]],
        "4": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称", "地点", "人员"],["反应堆状态"],["电站事件"]],
        "5": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称"], ["地点", "人员"],["反应堆状态"],["电站事件"]],
        "6": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称"], ["地点", "人员"],["反应堆状态","电站事件"]],
        "7": [["设备标识", "设备名称"],["系统标识", "系统名称"],["部件名称", "地点", "人员","反应堆状态","电站事件"]],
        "chatgpt0": [["地点"], ["人员"],["反应堆状态"],["系统名称", "系统标识"],["设备名称", "设备标识"],["部件名称"], ["电站事件"]]
    },
    "resume": {
        "chatgpt0": [['人名'], ['组织名'], ['地名'], ['国籍'], ['教育背景'], ['专业'], ['民族'], ['职称']],
        "chatgpt1": [['人名', '组织名','地名'], ['国籍', '民族'], ['教育背景','专业'], ['职称']]
    },
    "ontonotes4zh": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },
        "ontonotes4zh_300_42": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
        "0": [ ['地缘政治实体'], ['地名'], ['机构名称'],['人名']],
        "1": [['人名', '机构名称'], ['地缘政治实体','地名']],
    },
        "ontonotes4zh_300_52": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
        "1": [['人名', '机构名称'], ['地缘政治实体','地名']],
    },
        "ontonotes4zh_300_137": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
        "1": [['人名', '机构名称'], ['地缘政治实体','地名']],
    },
    "weibo": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
        "1": [['人名', '机构名称'], ['地缘政治实体','地名']],
    },    
    "msra_300_42": {
        "chatgpt0": [['人物'], ['地点'], ['机构']],
        "0": [['人物'], ['地点', '机构']],
        "1": [['人物'], ['机构', '地点']],
        "2": [['地点', '机构'],['人物']],
    },        
    "scierc": {
        "0": [["Task"],["Method"],["Evaluation Metric"],["Material"],["Other Scientific Terms"],["Generic"]],
        "1": [["Task","Method"],["Evaluation Metric","Material"],["Other Scientific Terms","Generic"]],
        "2": [["Method"],["Generic"],["Task"],["Material"],["Evaluation Metric"],["Other Scientific Terms"]],
        "chatgpt0": [["Material"],["Method"],["Task"],["Evaluation Metric"],["Other Scientific Terms"],["Generic"]],
        "chatgpt1": [["Material","Method","Task","Evaluation Metric"],["Other Scientific Terms","Generic"]],
    },
    "genia":{
        "0": [["protein"],["cell_type"],["cell_line"],["DNA"],["RNA"]],
        "chatgpt0": [["DNA","RNA"],["cell_type","protein"],["cell_line"]]
    },
        "genia_300_42":{
        "0": [["protein"],["cell_type"],["cell_line"],["DNA"],["RNA"]],
        "chatgpt0": [["DNA","RNA"],["cell_type","protein"],["cell_line"]]
    },
    "conll2003":{
        "0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
        "chatgpt0": [["Person"],["Organization"],["Location"],["Miscellaneous"]],
        "chatgpt1": [["Person","Organization","Location"],["Miscellaneous"]],
        "chatgpt2": [["Person","Organization"],["Location"],["Miscellaneous"]],
        "chatgpt3": [["Person","Location"],["Organization"],["Miscellaneous"]],
        "chatgpt4": [["Organization","Location"],["Person"],["Miscellaneous"]],
    },
        "conll2003_300_42":{
        "0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
        "1": [["Miscellaneous"],["Location"],["Organization"],["Person"]],
        "chatgpt0": [["Person"],["Organization"],["Location"],["Miscellaneous"]],
        "chatgpt1": [["Person","Organization","Location"],["Miscellaneous"]],
        "chatgpt2": [["Person","Organization"],["Location"],["Miscellaneous"]],
        "chatgpt3": [["Person","Location"],["Organization"],["Miscellaneous"]],
        "chatgpt4": [["Organization","Location"],["Person"],["Miscellaneous"]],
        "chatgpt5": [["Miscellaneous"],["Person","Organization","Location"]],
    },
     "conll2003_300_52":{
        "0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
     },
     "conll2003_300_137":{
        "0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
     },
            "conll2003pp_300_42":{
        "0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
        "1": [["Miscellaneous"],["Location"],["Organization"],["Person"]],
        "2": [["Person","Organization"],["Location","Miscellaneous"]],
        "chatgpt0": [["Person"],["Organization"],["Location"],["Miscellaneous"]],
        "chatgpt1": [["Person","Organization","Location"],["Miscellaneous"]],
        "chatgpt2": [["Person","Organization"],["Location"],["Miscellaneous"]],
        "chatgpt3": [["Person","Location"],["Organization"],["Miscellaneous"]],
        "chatgpt4": [["Organization","Location"],["Person"],["Miscellaneous"]],
        "chatgpt5": [["Miscellaneous"],["Person","Organization","Location"]],
    },
        "wnut17_300_42":{
            "0":[["Person"], ["Creative work"], ["Group"], ["Location"], ["Corporation"], ["Product"]],
            "1":[["Person"], ["Group"], ["Creative work"], ["Corporation"], ["Product"], ["Location"]],
            "2":[["Person"], ["Creative work"], ["Group"], ["Product"], ["Corporation"], ["Location"]],
            "3":[["Person","Group","Creative work"], ["Product","Corporation","Location"]],
            "chatgpt0":[["Person"], ["Location"], ["Corporation"], ["Product"], ["Creative work"], ["Group"]],
            "chatgpt1":[["Person","Location","Corporation"], ["Product","Creative work","Group"]],
        },
    "ace05en":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]]
    },
        "ace05en_300_42":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]],
        "0": [["Person","Organization"],["Facility","Weapon","Vehicle"],["Geo-Political Entity", "Location"]],
        "1": [["Person"],["Organization"],["Facility","Weapon","Vehicle"],["Geo-Political Entity", "Location"]],
        "2": [["Person"],["Organization"],["Geo-Political Entity", "Location"],["Facility","Weapon","Vehicle"]],
        "3": [["Person"],["Organization"],["Location"],["Geo-Political Entity"],["Facility","Weapon","Vehicle"]],
        "4": [["Person"],["Organization"],["Geo-Political Entity"],["Location"],["Facility","Weapon","Vehicle"]],
    },
        "ace05en_300_52":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]]
    },
        "ace05en_300_137":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]]
    },
     "ace04en":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]]
    },
    "ace04en_0_300_42":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]],
        "1": [["Person"],["Organization"],["Facility","Weapon","Vehicle"],["Geo-Political Entity", "Location"]],
        "2": [["Person"],["Organization"],["Geo-Political Entity", "Location"],["Facility","Weapon","Vehicle"]],
        "3": [["Person"],["Organization"],["Location"],["Geo-Political Entity"],["Facility","Weapon","Vehicle"]],
        "4": [["Person"],["Organization"],["Geo-Political Entity"],["Location"],["Facility","Weapon","Vehicle"]],
    },
    "ontonotes5en_300_42":{
        "chatgpt0": [["Person Name", "Location","Organization"],["Date", "Time", "Event"],["Geographical/Social/Political Entity", "Facility"],["Work of Art, Language"], ["Nationality/Other/Religion/Political", "Money", "Percent", "Quantity"],["Cardinal", "Ordinal", "Product"],["Law"]],
        "chatgpt1": [["Person Name"],["Location"],["Organization"],["Geographical/Social/Political Entity"],["Facility"],["Date"],["Time"],
                     ["Event"],["Work of Art"],["Language"],["Nationality/Other/Religion/Political"],["Money"],["Percent"],["Quantity"],
                     ["Cardinal"],["Ordinal"],["Product"],["Law"]],
        "0": [["Person Name"], ["Nationality/Other/Religion/Political"], ["Geographical/Social/Political Entity"], ["Organization"], ["Location","Facility","Product"],["Date", "Time", "Event"],["Percent","Quantity","Money"],["Ordinal","Cardinal"],["Work of Art","Law","Language"]]
    },
    "bc5cdr_300_42":{
        "chatgpt0": [["Chemical"], ["Disease"]],
        "0": [["Disease"], ["Chemical"]]
    },
        "bc5cdr_300_52":{
        "chatgpt0": [["Chemical"], ["Disease"]],
        "0": [["Disease"], ["Chemical"]]
    },
        "bc5cdr_300_137":{
        "chatgpt0": [["Chemical"], ["Disease"]],
        "0": [["Disease"], ["Chemical"]]
    },
    "JNLPBA_300_42":{
        "chatgpt0": [["protein"],["cell type"],["cell line"],["DNA"],["RNA"]],
        "chatgpt1": [["DNA","RNA"],["cell type","protein"],["cell line"]],
        "chatgpt2": [["DNA","RNA"],["cell type","cell line"],["protein"]], 
        "chatgpt3": [["protein"],["cell type","cell line"],["DNA","RNA"]],
    },
    "JNLPBA_300_52":{
        "chatgpt0": [["protein"],["cell type"],["cell line"],["DNA"],["RNA"]],
    },
    "JNLPBA_300_137":{
        "chatgpt0": [["protein"],["cell type"],["cell line"],["DNA"],["RNA"]],
    },
    "FEWNERD_300_42":{
        "chatgpt0": [["Person"],["Location"],["Organization"],["Building"],["Event"],["Product"], ["Art"], ["Other"]],
    },
        "wikigold":{
        "chatgpt0": [["Organization"],["Location"],["Person"],["Miscellaneous"]],
        },
    "bionlp11id_300_42":{
        # "chatgpt0": [['Chemical'], ['Organism'], ['Protein'], ['Regulon-operon']],
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
        },
    "bionlp11id_300_52":{
        # "chatgpt0": [['Chemical'], ['Organism'], ['Protein'], ['Regulon-operon']],
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
        },
    "bionlp11id_300_137":{
        # "chatgpt0": [['Chemical'], ['Organism'], ['Protein'], ['Regulon-operon']],
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
        },
    "bionlp13pc_300_42":{
        "chatgpt0": [['Simple_chemical'], ['Gene_or_gene_product'], ['Cellular_component'], ['Complex']],
        "chatgpt1": [['Simple_chemical'], ['Cellular_component'], ['Complex'], ['Gene_or_gene_product']],
        },
    "craft_300_42":{
        "chatgpt0": [['NCBI Taxonomy'], ['Gene or Gene Product'], ['Chemical Entities of Biological Interest'], ['Cell Ontology'], ['Gene Ontology'], ['Sequence Ontology']],
        },
    "craft_300_52":{
        "chatgpt0": [['NCBI Taxonomy'], ['Gene or Gene Product'], ['Chemical Entities of Biological Interest'], ['Cell Ontology'], ['Gene Ontology'], ['Sequence Ontology']],
        },
    "craft_300_137":{
        "chatgpt0": [['NCBI Taxonomy'], ['Gene or Gene Product'], ['Chemical Entities of Biological Interest'], ['Cell Ontology'], ['Gene Ontology'], ['Sequence Ontology']],
        }
}

my_openai_api_keys = [
    # ................ 第二个月 120 刀 ...................................................
    # "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", # 120, 0814_1, quota
    # "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", # 120, 0814_2, 
    # "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", # 120, 0820_1
    # "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", # 120, 0820_2
    # "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", # 120, 0820_3
    # "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", # 120, 0820_4, quota
    # "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", # 120, 0820_5, 
    # "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", # 120, 0820_6
    # "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", # 120, 0820_7, 
    # ...................120$...................................................
    # "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", # 120, 0904，崇宇信息 【每月限10刀】，转移组织到emmaxtyu
    # "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", # 120, 0904，崇宇信息 【每月限10刀】
    # "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", # 120, 0904，hstock
    # "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", # 120, 0904，hstock
    # "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", # 120, 0905，星星
    # "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", # 120, 0905，星星
    # {"key":"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", "org":"org-6wN5qqLfNs36ocwPAyg6ZeIj"}, # group2, 账号x4kacdvnw61dyof4ez@maiil10year.com
    # {"key":"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", "org":"org-yI4USMfxh9kUB1JjbhmKVGGd"},
    # {"key":"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", "org":"org-L1zKZ9z6G1AaVZ2Kird32xDm"},
    # {"key":"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", "org":"org-jKJWMltiAiiJFl0UPmfGmw8g"},
    # {"key":"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", "org":"org-DwE9apGBbPP8EjsKGmLjdxlK"}, # group3, 账号x4kacdvnw61dyof4ez@maiil10year.com
    # {"key":"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", "org":"org-VG8A4XpOIFPcDwS8Yrv38LgD"},
    # {"key":"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", "org":"org-9X2M6LkCEAKbWC2ReRYEPOw9"},
    # {"key":"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", "org":"org-T2vRThxejFSZOXjRHyIqa7NJ"},
    # ........................0909, hstock, 120$........................................
    # {"key":"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"},
    # ........................0910, xingxing, 120$........................................
    {"key":"sk-KKMhePrdAshu5ZbxC8849eAe3e334708BdDe6f4a99Bd2a0d", "set_base":True, "api_base":"https://api.ai-gaochao.cn/v1"},
    # {"key":"sk-fCL5PIMKAx0recCi665e3887285b4297A1Ff3bC57c419cE6", "set_base":True, "api_base":"https://api.ai-gaochao.cn/v1"},
]

# ports = [
#     # "8000",
#     # "8001",
#     # "8002"
#     "8100"
# ]