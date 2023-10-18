import sys
sys.path.append("code")
from utils import dict2json

class PromptPoolEnglish(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname
    
    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "You are an expert in named entity recognition. You are good at information extraction.\n"

        return task_hint
 
    def get_reason_hint(self, args):
        if args.reason_hint is None:
            return ""
        
        if args.reason_hint == "ToolUseTok":
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
         
        
    def get_task_instruction(self, args):
        label_set = "Given entity label set: %s\n" % (args.id2label)
        
        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "Given the text and the corresponding word segmentation, please recognize the named entities in the given text. "
            if args.tool_aug == "ToolPos":
                given = "Given the text and the corresponding Part-of-Speech tags, please recognize the named entities in the given text. "
            if args.tool_aug == "ToolDep":
                given = "Given the text and the corresponding dependency tree, please recognize the named entities in the given text. "     
            if args.tool_aug == "ToolCon":
                given = "Given the text and the corresponding constituency tree, please recognize the named entities in the given text. "                              
        else:
            given = "Please recognize the named entities in the given text. "
      
        ans_format = "Based on the given entity label set, provide answer in the following JSON format: [{\"Entity Name\": \"Entity Label\"}]. If there is no entity in the text, return the following empty list: []. "

        task_instruct = label_set + given + ans_format          
            
        return task_instruct            


    def get_prompt_prefix(self, args):
        task_hint = self.get_domain_hint(args)
        task_instruction = self.get_task_instruction(args)
        # where to put syntactic prompting, font or back
        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "f" else ""
        prefix = task_hint + task_instruction + reason_hint
        
        return prefix


    def get_prompt_for_demo(self, args, demo):
        demo_sent = demo["sentence"]
        demo_label = demo["label"]
        if isinstance(demo_label, str):
            demo_label = eval(demo_label)  
        if isinstance(demo_label, dict):
            demo_label = dict2json(demo_label)

        demo_prompt = "\nText: %s" % (demo_sent)

        if args.reason_hint is None and args.tool_aug is None:
            demo_prompt += "\nAnswer: %s" % demo_label
            return demo_prompt

        if not (args.reason_hint is None) and args.tool_aug is None:
            if args.reason_hint in ["pos_conj"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\nAnswer: \nPart-of-Speech tagging: %s\nEntities: %s" % (demo_pos, demo_label)           
                return demo_prompt  
            
            if args.reason_hint in ["con_conj"]:
                demo_con = demo["con_str"]
                demo_prompt += "\nAnswer: \nConstituency parsing: %s\nEntities%s" % (demo_con, demo_label)           
                return demo_prompt
            
            if args.reason_hint in ["dep_conj"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nAnswer: \Dependency parsing: %s\nEntities%s" % (demo_dep, demo_label)
                return demo_prompt

        if args.reason_hint is None and not (args.tool_aug is None):
            if args.tool_aug in ["ToolPos"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\nPart-of-Speech tags: %s\nAnswer: %s" % (demo_pos, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolDep"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nDependency Tree: %s\nAnswer: %s" % (demo_dep, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolCon"]:
                demo_con = demo["con_str"]
                demo_prompt += "\nConstituency Tree: %s\nAnswer: %s" % (demo_con, demo_label)
                return demo_prompt                     
        
        if not (args.reason_hint is None) and not (args.tool_aug is None):
            if args.reason_hint in ["pos_conj"] and args.tool_aug=="ToolPos":
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\nPart-of-Speech tags: %s\nAnswer: %s" % (demo_pos, demo_label)        
                return demo_prompt
            
            if args.reason_hint in ["con_conj"] and args.tool_aug=="ToolCon":
                demo_con = demo["con_str"]
                demo_prompt += "\nConstituency Tree: %s\nAnswer: %s" % (demo_con, demo_label)
                return demo_prompt    

            if args.reason_hint in ["dep_conj"] and args.tool_aug=="ToolDep":
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nDependency Tree: %s\nAnswer: %s" % (demo_dep, demo_label)
                return demo_prompt              

        
    def get_prompt_postfix(self, args, query):
        sent = query["sentence"]

        if args.tool_aug:
            if args.tool_aug == "ToolTokCoarse":
                tok_coarse = query["tok/coarse"] 
                input_output_instruction = "\nText: %s\nTokenization: %s\nAnswer: " % (sent, tok_coarse)
            elif args.tool_aug == "ToolPos":
                pos = query["tok_pos_pair_str"]   
                input_output_instruction = "\nText: %s\nPart-of-Speech tags: %s\nAnswer: " % (sent, pos)                    
            elif args.tool_aug == "ToolDep":
                dep = query["trip_dep"]   
                input_output_instruction = "\nText: %s\nDependency tree: %s\nAnswer: " % (sent, dep)    
            elif args.tool_aug == "ToolCon":
                dep = query["con_str"]   
                input_output_instruction = "\nText: %s\Consitituency tree: %s\nAnswer: " % (sent, dep)                   
            else:
                raise ValueError(f"Unrecognized tool_aug: {args.tool_aug}")                                        
        else:
                input_output_instruction = "\nText: %s\nAnswer: " % (sent)

        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "b" else ""
        postfix = input_output_instruction + reason_hint
        return postfix



class PromptPoolChinese(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname

    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "你是命名实体识别方面的专家。你很擅长信息抽取。\n"

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
        
        if args.reason_hint == "pos_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行词性标注。接着，我们基于标注的词性识别命名实体。"
            else:
                reason_hint = "首先，你应该进行词性标注。接着，你应该基于标注的词性识别命名实体。"
            return reason_hint
        
        if args.reason_hint == "con_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行成分句法解析。接着，我们基于成分树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行成分句法解析。接着，你应该基于成分树识别命名实体。"
            return reason_hint      

        if args.reason_hint == "dep_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行依存句法解析。接着，我们基于依存树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行依存句法解析。接着，你应该基于依存树识别命名实体。"
            return reason_hint
        
        if args.reason_hint in ["tok_conj", "tok_fine_conj", "tok_coarse_conj"]:
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行分词。接着，我们基于分词结果识别命名实体。"
            else:
                reason_hint = "首先，你应该进行分词。接着，你应该基于分词结果识别命名实体。"
            return reason_hint    
    

    def get_task_instruction(self, args):
        label_set = "给定实体标签集：%s\n" % (args.id2label)

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "给定文本和对应的分词结果，请识别文本中的命名实体。"
            elif args.tool_aug == "ToolPos":
                given = "给定文本和对应的词性标注，请识别文本中的命名实体。"
            elif args.tool_aug == "ToolDep":
                given = "给定文本和对应的依存树，请识别文本中的命名实体。"   
            elif args.tool_aug == "ToolCon":
                given = "给定文本和对应的成分树，请识别文本中的命名实体。"
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")                                
        else:
            given = "请识别给定文本中的命名实体。"

        ans_format = "基于给定的实体标签集，以如下JSON格式提供答案：[{\"实体名称\": \"实体标签\"}]。如果文本中没有实体，请返回如下空列表：[]。"
            
        task_instruct = label_set + given + ans_format      

        return task_instruct    

    def get_prompt_prefix(self, args):
        task_hint = self.get_domain_hint(args)
        task_instruction = self.get_task_instruction(args)
        # where to put syntactic prompting, font or back
        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "f" else ""
        prefix = task_hint + task_instruction + reason_hint
        
        return prefix

    def get_prompt_for_demo(self, args, demo):
        demo_sent = demo["sentence"]
        demo_label = demo["label"]
        if isinstance(demo_label, str):
            demo_label = eval(demo_label)  
        if isinstance(demo_label, dict):
            demo_label = dict2json(demo_label)

        demo_prompt = "\n文本：%s" % (demo_sent)

        if args.reason_hint is None and args.tool_aug is None:
            demo_prompt += "\n答案：%s" % demo_label
            return demo_prompt
        
        if not (args.reason_hint is None) and args.tool_aug is None:
            if args.reason_hint in ["pos_conj"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\n答案：\n词性标注：%s\n命名实体：%s" % (demo_pos, demo_label)           
                return demo_prompt  
            
            if args.reason_hint in ["con_conj"]:
                demo_con = demo["con_str"]
                demo_prompt += "\n答案：\n成分句法分析：%s\n命名实体：%s" % (demo_con, demo_label)           
                return demo_prompt    

            if args.reason_hint in ["dep_conj"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n答案：\n依存分析：%s\n命名实体：%s" % (demo_dep, demo_label)
                return demo_prompt              

            if args.reason_hint in ["tok_coarse_conj"]:
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n答案：\n分词：%s\n命名实体：%s" % (demo_tok_coarse, demo_label)           
                return demo_prompt

        if args.reason_hint is None and not (args.tool_aug is None):
            if args.tool_aug in ["ToolTokCoarse"]:
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n分词：%s\n答案：%s" % (demo_tok_coarse, demo_label)
                return demo_prompt
            
            if args.tool_aug in ["ToolPos"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\n词性标注：%s\n答案：%s" % (demo_pos, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolDep"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n依存树：%s\n答案：%s" % (demo_dep, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolCon"]:
                demo_con = demo["con_str"]
                demo_prompt += "\n成分树：%s\n答案：%s" % (demo_con, demo_label)
                return demo_prompt
        
        if not (args.reason_hint is None) and not (args.tool_aug is None):
            if args.reason_hint =="ToolUseTok" and args.tool_aug=="ToolTokCoarse":
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n分词：%s\n答案：%s" % (demo_tok_coarse, demo_label)         
                return demo_prompt
        
            if args.reason_hint =="ToolUsePos" and args.tool_aug=="ToolPos":
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\n词性标注：%s\n答案：%s" % (demo_pos, demo_label)          
                return demo_prompt  

            if args.reason_hint == "ToolUseDep" and args.tool_aug=="ToolDep":
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n依存树：%s\n答案：%s" % (demo_dep, demo_label)        
                return demo_prompt
            
            if args.reason_hint == "ToolUseCon" and args.tool_aug=="ToolCon":
                demo_con = demo["con_str"]
                demo_prompt += "\n成分树：%s\n答案：%s" % (demo_con, demo_label)         
                return demo_prompt
        
    def get_prompt_postfix(self, args, query):
        sent = query["sentence"]

        if args.tool_aug:
            if args.tool_aug == "ToolTokCoarse":
                tok_coarse = query["tok/coarse"]
                input_output_instruction = "\n文本：%s\n分词：%s\n答案：" % (sent, tok_coarse)
            elif args.tool_aug == "ToolPos":
                pos = query["tok_pos_pair_str"]  
                input_output_instruction = "\n文本：%s\n词性标注：%s\n答案：" % (sent, pos)
            elif args.tool_aug == "ToolDep":
                dep = query["trip_dep"]
                input_output_instruction = "\n文本：%s\n依存树：%s\n答案：" % (sent, dep)    
            elif args.tool_aug == "ToolCon":
                dep = query["con_str"]
                input_output_instruction = "\n文本：%s\n成分树：%s\n答案：" % (sent, dep)
            else:
                raise ValueError(f"Unrecognized tool_aug: {args.tool_aug}")
        else:
                input_output_instruction = "\n文本：%s\n答案：" % (sent)

        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "b" else ""
        postfix = input_output_instruction + reason_hint
        return postfix
