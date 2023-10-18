dataname=msra_5_samples

# syntactic prompting
reason_hint=pos_conj
# back
reason_hint_person=first
reason_hint_pos=b

order=chatgpt0

start_time=10172200

python code/qa_dialogue/GeneratePrompts.py \
        --dataname $dataname \
        --order $order \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

python code/qa_dialogue/AskGPT.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

python code/qa_dialogue/ComputeMetric.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

# front
reason_hint_person=second
reason_hint_pos=f
python code/qa_dialogue/GeneratePrompts.py \
        --dataname $dataname \
        --order $order \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

python code/qa_dialogue/AskGPT.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos

python code/qa_dialogue/ComputeMetric.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \
        --reason_hint $reason_hint --reason_hint_person $reason_hint_person --reason_hint_pos $reason_hint_pos
