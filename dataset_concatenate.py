from datasets import load_dataset, load_from_disk, concatenate_datasets
import os


# train_dataset = load_from_disk("./cache/wikitext/refer_dataset")
# dataset = load_from_disk("./cache/wikitext/refer_dataset")
# concatenated_dataset = concatenate_datasets(train_dataset, dataset)

dataset_dir = "./cache/wikitext/refer_dataset_gptj"
concatenated_dataset = None

for file_name in os.listdir(dataset_dir):
    data_path = os.path.join(dataset_dir, file_name)

    if concatenated_dataset is None:
        concatenated_dataset = load_from_disk(data_path)
    else:
        dataset = load_from_disk(data_path)
        concatenated_dataset = concatenate_datasets([concatenated_dataset, dataset])

concatenated_dataset.save_to_disk("./cache/wikitext/refer_dataset_gptj")