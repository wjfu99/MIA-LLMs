

accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/gpt2/wiki2/target/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 1e-4 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=0 --train_end_idx=6000 --eval_sta_idx=0 --eval_end_idx=600

accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/reference_model_v2 \
--block_size 128 --eval_steps 10 --save_epochs 5 --log_steps 100 \
-d wikitext -dc wikitext-2-raw-v1 -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 5e-5 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=6000 --train_end_idx=12000 --eval_sta_idx=600 --eval_end_idx=1200


# ag_news
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/gpt2/ag_news/target/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 1e-4 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000

# refer prompt
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/gpt2/ag_news/refer/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 5e-5 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000

# refer candidate
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/gpt2/ag_news/candidate/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m gpt2 --packing --use_dataset_cache \
-e 200 -b 8 -lr 5e-5 --gradient_accumulation_steps 8 \
--disable_lora --train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000