from datasets import load_dataset, Dataset
import os

class Private_Info_Dataset():
    def __init__(self, datafile: str, seed = 42, train_split = True, train_end_p = .75, eval_start_p = .75, enable_splitting=True):
        """All instances of private info dataset should contain raw_dataset with fields 'defender_private_information' 'attacker_target_information'."""
        import random
        import json
        with open(datafile, "r") as f:
            raw_dataset = json.load(f)
        self.raw_dataset = raw_dataset
        self.datafile = datafile

        # Shuffle
        random.seed(seed)
        random.shuffle(self.raw_dataset)

        # Split dataset
        if enable_splitting:
            train_end_index = int(len(self.raw_dataset) * train_end_p)
            eval_start_index = int(len(self.raw_dataset) * eval_start_p)
            if train_split:
                self.raw_dataset = self.raw_dataset[:train_end_index]
            else:
                self.raw_dataset = self.raw_dataset[eval_start_index:]
        
    def get_hf_dataset(self):
        return Dataset.from_list(self.raw_dataset)

        
def load_custom_dataset(dataset, seed, use_train_split=True, train_end_p=0.75, eval_start_p=0.75):
    if dataset == "three_layered_dataset":
        dataset_built = Private_Info_Dataset("datasets_directory/final_datasets/three_layered_dataset.json", seed=seed, train_split=use_train_split, train_end_p=train_end_p, eval_start_p=eval_start_p).get_hf_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataset_built