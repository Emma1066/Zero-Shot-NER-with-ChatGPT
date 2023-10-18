dataname=msra_5_samples

# tool augmentation
tool_aug=ToolPos
# syntactic prompting
reason_hint=ToolUsePos
# back
reason_hint_person=first
reason_hint_pos=b

start_time=10171900

# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

# front
reason_hint_person=second
reason_hint_pos=f
# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
