from datasets import load_dataset, load_from_disk, concatenate_datasets
import os


# train_dataset = load_from_disk("./cache/wikitext/refer_dataset")
# dataset = load_from_disk("./cache/wikitext/refer_dataset")
# concatenated_dataset = concatenate_datasets(train_dataset, dataset)

dataset_dir = "/mnt/data0/fuwenjie/MIA-LLMs/cache/ag_news/None/refer@decapoda-research/llama-7b-hf"
concatenated_dataset = None

for file_name in os.listdir(dataset_dir):
    data_path = os.path.join(dataset_dir, file_name)
    if os.path.isdir(data_path):
        if concatenated_dataset is None:
            concatenated_dataset = load_from_disk(data_path)
        else:
            dataset = load_from_disk(data_path)
            concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])

concatenated_dataset.save_to_disk("/mnt/data0/fuwenjie/MIA-LLMs/cache/ag_news/None/refer@decapoda-research/llama-7b-hf")