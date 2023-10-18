dataname=msra_5_samples

start_time=1015

# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \

