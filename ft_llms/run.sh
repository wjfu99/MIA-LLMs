accelerate launch llms_finetune.py \
--block_size 1024 --eval_steps 100 --save_steps 100 \
 -tf train.csv -vf validation.csv -m EleutherAI/gpt-j-6B \
 -b 1 --log_steps 100 -lr 5e-6 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18636 --disable_lora

accelerate launch llms_finetune.py \
--block_size 1024 --eval_steps 100 --save_steps 100 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 \
-e 20 -b 1 -lr 5e-6 --gradient_accumulation_steps 16 \
 --disable_lora