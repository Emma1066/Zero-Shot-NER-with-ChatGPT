class PromptPoolEnglish(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname
    
    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "You are an expert in named entity recognition. You are good at information extraction."

        return task_hint
 
    def get_reason_hint(self, args):
        if args.reason_hint is None:
            return ""
        
        if args.reason_hint == "ToolUseTokCoarse":
            if args.reason_hint_person == "first":
                reason_hint = "Let's infer named entities step by step from the text based on the given word segmentation."
            else:
                reason_hint = "Please infer named entities step by step from the text based on the given word segmentation."
            return reason_hint       
        
        if args.reason_hint == "ToolUsePos":
            if args.reason_hint_person == "first":
                reason_hint = "Let's infer named entities step by step from the text based on the given Part-of-Speech tags."
            else:
                reason_hint = "Please infer named entities step by step from the text based on the given Part-of-Speech tags."
            return reason_hint   

        if args.reason_hint == "ToolUseCon":
            if args.reason_hint_person == "first":
                reason_hint = "Let's infer named entities step by step from the text based on the given constituency tree."
            else:
                reason_hint = "Please infer named entities step by step from the text based on the given constituency tree."
            return reason_hint       

        if args.reason_hint == "ToolUseDep":
            if args.reason_hint_person == "first":
                reason_hint = "Let's infer named entities step by step from the text based on the given dependency tree."
            else:
                reason_hint = "Please infer named entities step by step from the text based on the given dependency tree."
            return reason_hint                       
        
        if args.reason_hint == "noun_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's recognize the noun phrases. Then, we recognize named entities based on the noun phrases. "
            else:
                reason_hint = "First, you should recognize the noun phrases. Then, you should recognize named entities based on the noun phrases. "
            return reason_hint
        
        if args.reason_hint == "pos_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform Part-of-Speech tagging. Then, we recognize named entities based on the Part-of-Speech tags. "
            else:
                reason_hint = "First, you should perform Part-of-Speech tagging. Then, you should recognize named entities based on the Part-of-Speech tags. "
            return reason_hint
        
        if args.reason_hint == "dep_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform dependency parsing. Then, we recognize named entities based on the dependency tree. "
            else:
                reason_hint = "First, you should perform dependency parsing. Then, you should recognize named entities based on the dependency tree. "
            return reason_hint

        if args.reason_hint == "con_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform constituency parsing. Then, we recognize named entities based on the constituency tree. "
            else:
                reason_hint = "First, you should perform constituency parsing. Then, you should recognize named entities based on the constituency tree. "
            return reason_hint
    
    def get_task_desc(self, args):
        label_set = "Given entity label set: %s" % (args.id2label)
        
        task_require = "Based on the given entity label set, please recognize the named entities in the given text."

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                task_require = "Given the text and the corresponding word segmentation, please recognize the named entities in the given text based on the entity label set. "
            elif args.tool_aug == "ToolPos":
                task_require = "Given the text and the corresponding Part-of-Speech tags, please recognize the named entities in the given text. "
            elif args.tool_aug == "ToolDep":
                task_require = "Given the text and the corresponding dependency tree, please recognize the named entities in the given text. "
            elif args.tool_aug == "ToolCon":
                task_require = "Given the text and the corresponding constituency tree, please recognize the named entities in the given text. "
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")      

        task_desc = label_set + "\n" + task_require

        return task_desc
    
    def get_query_info(self, args, query):
        given = "Text: %s" % (query["sentence"])

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "Text: %s\nWord segmentation: %s" % (query["sentence"], query["tok/coarse"])
            elif args.tool_aug == "ToolPos":
                given = "Text: %s\nPart-of-Speech tags: %s" % (query["sentence"], query["tok_pos_pair_str"])
            elif args.tool_aug == "ToolDep":
                given = "Text: %s\nDependency tree: %s" % (query["sentence"], query["trip_dep"])
            elif args.tool_aug == "ToolCon":
                given = "Text: %s\nConstituency tree: %s" % (query["sentence"], query["con_str"])
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")
            
        return given

    def get_output_constraint(self, form="json"):
        if form == "json":
            output_constraint = "Provide the answer in the following JSON format: [{\"Entity Name\": \"Entity Label\"}]. If there is no corresponding entity, return the following empty list: []."
        elif form == "list":
            output_constraint = "Provide the answer in the following list format: [\"Entity Name 1\", \"Entity Name 2\", ...]. If there is no corresponding entity, return the following empty list: []."
        else:
            raise ValueError(f"Unrecognized form = {form}")
        return output_constraint
    
    def get_answer_hint(self):
        answer_hint = "Answer: "
        return answer_hint
    
    def get_question_hint(self):
        question_hint = "Question: "
        return question_hint

    def get_question(self, args, target_types):
        target_types = [f"\"{x}\"" for x in target_types]
        if len(target_types) == 1:
            target_types_str = target_types[0]
        else:
            target_types_str = ", ".join(target_types[:-1])
            target_types_str += " and %s" % (target_types[-1])

        question_require = "What are the named entities labeled as %s in the text? " % target_types_str
        question = question_require  

        return question

    def fetch_target_type_entity(self, args, demos, target_type):
        target_entities = []
        for demo in demos:
            demo_label = demo["label"]
            if isinstance(demo_label, str):
                demo_label = eval(demo_label)
            for ment in demo_label:
                if demo_label[ment] == target_type:
                    target_entities.append(ment)
        
        return target_entities
    

class PromptPoolChinese(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname

    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "你是命名实体识别方面的专家。你很擅长信息抽取。"

        return task_hint
    
    def get_reason_hint(self, args):
        if args.reason_hint is None:
            return ""
        
        if args.reason_hint == "ToolUseTok":
            if args.reason_hint_person == "first":
                reason_hint = "让我们基于给定的分词结果，从文本一步步推理出命名实体。"
            else:
                reason_hint = "请基于给定的分词结果，从文本一步步推理出命名实体。"
            return reason_hint
        
        if args.reason_hint == "ToolUsePos":
            if args.reason_hint_person == "first":
                reason_hint = "让我们基于给定的词性标注，从文本一步步推理出命名实体。"                
            else:
                reason_hint = "请基于给定的词性标注，从文本一步步推理出命名实体。"                
            return reason_hint

        if args.reason_hint == "ToolUseCon":
            if args.reason_hint_person == "first":
                reason_hint = "让我们基于给定的成分树，从文本一步步推理出命名实体。"                 
            else:
                reason_hint = "请基于给定的成分树，从文本一步步推理出命名实体。"                   
            return reason_hint       

        if args.reason_hint == "ToolUseDep":
            if args.reason_hint_person == "first":
                reason_hint = "让我们基于给定的依存树，从文本一步步推理出命名实体。"                 
            else:
                reason_hint = "请基于给定的依存树，从文本一步步推理出命名实体。"                          
            return reason_hint
        
        if args.reason_hint == "noun_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们识别名词。接着，我们基于名词识别命名实体。"
            else:
                reason_hint = "首先，你应该识别名词。接着，你应该基于名词识别命名实体。"
            return reason_hint

        if args.reason_hint == "tok_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行分词。接着，我们基于分词结果识别命名实体。"
            else:
                reason_hint = "首先，你应该进行分词。接着，你应该基于分词结果识别命名实体。"
            return reason_hint
        
        if args.reason_hint == "pos_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行词性标注。接着，我们基于标注的词性识别命名实体。"
            else:
                reason_hint = "首先，你应该进行词性标注。接着，你应该基于标注的词性识别命名实体。"
            return reason_hint

        if args.reason_hint == "dep_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行依存句法解析。接着，我们基于依存树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行依存句法解析。接着，你应该基于依存树识别命名实体。"
            return reason_hint    

        if args.reason_hint == "con_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行成分句法解析。接着，我们基于成分树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行成分句法解析。接着，你应该基于成分树识别命名实体。"
            return reason_hint     

    def get_task_desc(self, args):
        label_set = "给定实体标签集：%s" % (args.id2label)

        task_require = "请基于给定的实体标签集，识别给定文本中的命名实体。"

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                task_require = "给定文本和对应的分词结果，请基于实体标签集识别文本中的命名实体。"
            elif args.tool_aug == "ToolPos":
                task_require = "给定文本和对应的词性标注，请基于实体标签集识别文本中的命名实体。"
            elif args.tool_aug == "ToolDep":
                task_require = "给定文本和对应的依存树，请基于实体标签集识别文本中的命名实体。"
            elif args.tool_aug == "ToolCon":
                task_require = "给定文本和对应的成分树，请基于实体标签集识别文本中的命名实体。"
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")      

        task_desc = label_set + "\n" + task_require

        return task_desc
    
    def get_query_info(self, args, query):
        given = "文本：%s" % (query["sentence"])

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "文本：%s\n分词：%s" % (query["sentence"], query["tok/coarse"])
            elif args.tool_aug == "ToolPos":
                given = "文本：%s\n词性标注：%s" % (query["sentence"], query["tok_pos_pair_str"])
            elif args.tool_aug == "ToolDep":
                given = "文本：%s\n依存树：%s" % (query["sentence"], query["trip_dep"])  
            elif args.tool_aug == "ToolCon":
                given = "文本：%s\n成分树：%s" % (query["sentence"], query["con_str"])          
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")      
            
        return given
    
    def get_output_constraint(self, form="json"):
        if form == "json":
            output_constraint = "请以如下JSON格式提供答案：[{\"实体名称\": \"实体标签\"}]。如果没有对应实体，请返回如下空列表：[]。"
        elif form == "list":
            output_constraint = "请以如下列表格式提供答案：[\"实体名称1\", \"实体名称2\", ...]。如果没有对应实体，请返回如下空列表：[]。"
        else:
            raise ValueError(f"Unrecognized form = {form}")
        return output_constraint
    
    def get_answer_hint(self):
        answer_hint = "答案："
        return answer_hint
    
    def get_question_hint(self):
        question_hint = "问题："
        return question_hint

    def get_question(self, args, target_types):
        target_types = [f"\"{x}\"" for x in target_types]
        if len(target_types) == 1:
            target_types_str = target_types[0]
        else:
            target_types_str = "、".join(target_types[:-1])
            target_types_str += "和%s" % (target_types[-1])

        question_require = "文本中标签为%s的实体有哪些？" % target_types_str
        question = question_require

        return question
    
    def fetch_target_type_entity(self, args, demos, target_type):
        target_entities = []
        for demo in demos:
            demo_label = demo["label"]
            if isinstance(demo_label, str):
                demo_label = eval(demo_label)
            for ment in demo_label:
                if demo_label[ment] == target_type:
                    target_entities.append(ment)
        
        return target_entities    