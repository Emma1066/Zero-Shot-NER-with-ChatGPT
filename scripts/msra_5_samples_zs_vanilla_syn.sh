dataname=msra_5_samples

# syntactic prompting
reason_hint=pos_conj
# back
reason_hint_person=first
reason_hint_pos=b

start_time=10171900
# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

# front
reason_hint_person=second
reason_hint_pos=f
start_time=10171901
# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos