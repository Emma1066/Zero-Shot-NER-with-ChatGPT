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
    "msra_5_samples": "zh",
    "weibo": "zh",
    "msra": "zh",
    "msra_300_42": "zh",
    "ontonotes4zh": "zh",
    "ontonotes4zh_300_42": "zh",
    "ontonotes4zh_300_52": "zh",
    "ontonotes4zh_300_137": "zh",
    "ace05en": "en",
    "ace05en_300_42": "en",
    "ace05en_300_52": "en",
    "ace05en_300_137": "en",
    "ace04en": "en",
    "ace04en_0_300_42": "en",
    "conll2003": "en",
    "conll2003_300_42": "en",
    "conll2003_300_52": "en",
    "conll2003_300_137": "en",
    "wnut17_300_42": "en",
    "bc5cdr": "en",
    "bc5cdr_300_42": "en",
    "bc5cdr_300_52": "en",
    "bc5cdr_300_137": "en",
    "bionlp11id_300_42": "en",
    "bionlp11id_300_52": "en",
    "bionlp11id_300_137": "en",
    "craft_300_42": "en",
    "craft_300_52": "en",
    "craft_300_137": "en",
}

dataset_label_order_map = {
    "msra_5_samples": {
        "chatgpt0": [['人物'], ['地点'], ['机构']],
    },   
    "weibo": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },    
    "msra_300_42": {
        "chatgpt0": [['人物'], ['地点'], ['机构']],
    },      
    "ontonotes4zh": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },
    "ontonotes4zh_300_42": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },
    "ontonotes4zh_300_52": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },
    "ontonotes4zh_300_137": {
        "chatgpt0": [['人名'], ['地名'], ['机构名称'], ['地缘政治实体']],
    },
    "ace05en":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]]
    },
    "ace05en_300_42":{
        "chatgpt0": [["Person"],["Organization"],["Location"],["Facility"],["Weapon"],["Vehicle"],["Geo-Political Entity"]],
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
    },
    "conll2003":{
        "chatgpt0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
    },
    "conll2003_300_42":{
        "chatgpt0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
    },
    "conll2003_300_52":{
        "chatgpt0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
    },
    "conll2003_300_137":{
        "chatgpt0": [["Location"],["Organization"],["Person"],["Miscellaneous"]],
    },
    "wnut17_300_42":{
        "chatgpt0":[["Person"], ["Location"], ["Corporation"], ["Product"], ["Creative work"], ["Group"]],
    },
    "bc5cdr_300_42":{
        "chatgpt0": [["Chemical"], ["Disease"]],
    },
    "bc5cdr_300_52":{
        "chatgpt0": [["Chemical"], ["Disease"]],
    },
    "bc5cdr_300_137":{
        "chatgpt0": [["Chemical"], ["Disease"]],
    },
    "bionlp11id_300_42":{
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
    },
    "bionlp11id_300_52":{
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
    },
    "bionlp11id_300_137":{
        "chatgpt1": [['Protein'], ['Organism'],['Chemical'],  ['Regulon-operon']],
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

# Fill in your own OpenAI API keys. 
# If your key does not need to set API base, set "set_base" to False. Otherwise, set "set_base" to True, and fill the "api_base" with your corresponding base.
my_openai_api_keys = [
    {"key":"Your_API_Key", "set_base":False, "api_base":"Your_API_base"},
]
