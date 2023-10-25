
# ag_news
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/falcon/ag_news/target/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m tiiuae/falcon-7b --packing --use_dataset_cache \
-e 40 -b 4 -lr 5e-4 --gradient_accumulation_steps 16 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000

# refer candidate
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/falcon/ag_news/candidate/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d JulesBelveze/tldr_news -m tiiuae/falcon-7b --packing --use_dataset_cache \
-e 40 -b 4 -lr 1e-4 --gradient_accumulation_steps 8 \
--train_sta_idx=0 --train_end_idx=4767 --eval_sta_idx=0 --eval_end_idx=538

# refer oracle
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/falcon/ag_news/oracle/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m tiiuae/falcon-7b --packing --use_dataset_cache \
-e 40 -b 4 -lr 1e-4 --gradient_accumulation_steps 8 \
--train_sta_idx=10000 --train_end_idx=20000 --eval_sta_idx=1000 --eval_end_idx=2000

accelerate launch refer_data_generate.py

# refer prompt
accelerate launch ./ft_llms/llms_finetune.py --refer \
--output_dir ./ft_llms/falcon/ag_news/refer/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d ag_news -m tiiuae/falcon-7b --packing --use_dataset_cache \
-e 40 -b 4 -lr 5e-5 --gradient_accumulation_steps 8 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000