dataname=msra_5_samples

order=chatgpt0

start_time=10172200

python code/qa_dialogue/GeneratePrompts.py \
        --dataname $dataname \
        --order $order \

python code/qa_dialogue/AskGPT.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time

python code/qa_dialogue/ComputeMetric.py \
        --dataname $dataname \
        --order $order \
        --start_time $start_time \

