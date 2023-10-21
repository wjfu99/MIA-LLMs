#accelerate launch llms_finetune.py \
#--block_size 1024 --eval_steps 100 --save_steps 100 \
# -tf train.csv -vf validation.csv -m EleutherAI/gpt-j-6B \
# -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --disable_lora

accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/target_model \
--block_size 128 --eval_steps 100 --save_epochs 5 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 1e-4 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=0 --train_end_idx=6000 --eval_sta_idx=0 --eval_end_idx=600

accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/reference_model_v2 \
--block_size 128 --eval_steps 10 --save_epochs 5 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 5e-5 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=6000 --train_end_idx=12000 --eval_sta_idx=600 --eval_end_idx=1200


# start from gptj, save via step policy.
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/target_model_gptj \
--block_size 128 --eval_steps 20 --save_epochs 200 --log_steps 20 \
-d wikitext -dc wikitext-2-raw-v1 -m EleutherAI/gpt-j-6B --packing --use_dataset_cache \
-e 200 -b 8 -lr 1e-4 --gradient_accumulation_steps 8 \
--train_sta_idx=0 --train_end_idx=6000 --eval_sta_idx=0 --eval_end_idx=600

accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/reference_model_gptj \
--block_size 128 --eval_steps 20 --save_epochs 20 --log_steps 20 \
-d wikitext -dc wikitext-2-raw-v1 -m EleutherAI/gpt-j-6B --packing --use_dataset_cache \
-e 200 -b 8 -lr 5e-5 --gradient_accumulation_steps 8 \
--train_sta_idx=0 --train_end_idx=6000 --eval_sta_idx=0 --eval_end_idx=600


accelerate launch llms_finetune.py \
--block_size 1024 --eval_steps 100 --save_steps 100 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 \
-e 200 -b 1 -lr 5e-4 --gradient_accumulation_steps 8 \
 --disable_lora


accelerate launch llms_finetune.py \
--block_size 1024 --eval_steps 100 --save_steps 100 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 \
-e 100 -b 1 -lr 1e-3 --gradient_accumulation_steps 8 \
 --disable_lora

accelerate launch llms_finetune.py \
--block_size 1024 --eval_steps 100 --save_steps 100 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 \
-e 200 -b 1 -lr 1e-4 --gradient_accumulation_steps 8