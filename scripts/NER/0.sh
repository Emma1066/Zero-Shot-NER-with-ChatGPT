dataname=weibo
datamode=test

# reason_hint=ToolUseDep
# reason_hint_person=first
# reason_hint_pos=b
# tool_aug=ToolDep

order=1

few_shot_setting=zs

start_time=09052200

python OPENAI/code/NER/en_zh_qa_dialogue/GeneratePrompts.py \
        --dataname $dataname \
        --datamode $datamode \
        --order $order \
        --few_shot_setting $few_shot_setting \

python OPENAI/code/NER/en_zh_qa_dialogue/AskGPT.py \
        --dataname $dataname \
        --datamode $datamode \
        --order $order \
        --few_shot_setting $few_shot_setting \
        --start_time $start_time

python OPENAI/code/NER/en_zh_qa_dialogue/ComputeMetric.py \
        --dataname $dataname \
        --datamode $datamode \
        --order $order \
        --few_shot_setting $few_shot_setting \
        --start_time $start_time \

