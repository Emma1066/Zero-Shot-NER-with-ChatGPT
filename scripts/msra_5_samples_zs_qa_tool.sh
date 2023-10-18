dataname=msra_5_samples

# tool augmentation
tool_aug=ToolPos

order=chatgpt0

start_time=10172200

python code/qa_dialogue/GeneratePrompts.py \
        --dataname $dataname \
        --order $order \
        --tool_aug $tool_aug \

python code/qa_dialogue/AskGPT.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --tool_aug $tool_aug \

python code/qa_dialogue/ComputeMetric.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --tool_aug $tool_aug \

