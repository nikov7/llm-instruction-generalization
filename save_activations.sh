python save_LLMs_activations.py \
  --data_path=data/ \
  --input_file=ifeval_simple_v1.jsonl \
  --model_name_hf=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task_type=ifeval_simple_v1 \
  --hf_use_auth_token HF_TOKEN
  
python save_LLMs_activations.py \
  --data_path=data/ \
  --input_file=ifeval_simple_v2.jsonl \
  --model_name_hf=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task_type=ifeval_simple_v2 \
  --hf_use_auth_token HF_TOKEN
  
python save_LLMs_activations.py \
  --data_path=data/ \
  --input_file=ifeval_simple_v3.jsonl \
  --model_name_hf=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task_type=ifeval_simple_v3 \
  --hf_use_auth_token HF_TOKEN
