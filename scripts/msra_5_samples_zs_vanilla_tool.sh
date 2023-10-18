dataname=msra_5_samples

# tool augmentation
tool_aug=ToolPos

start_time=10171900

# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
        --tool_aug $tool_aug
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \
        --tool_aug $tool_aug