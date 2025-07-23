import os
import json
import torch
import datasets
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LabelFilter:
    def __init__(self, label_field):
        self.label_field = label_field

    def __call__(self, example):
        for label in self.label_field:       # if there is one of the label is not -1, keep it
            if label in example and isinstance(example[label], torch.Tensor):
                if len(example[label]) > 0 and not (example[label] == -1).all().item():
                    return True
        return ('dialogue_a' in example and 'dialogue_b' in example and      # if there is one of the dialogue data, keep 
                example['dialogue_a'] and example['dialogue_b'])

def load_datasets(tokenizer, dataset_paths, label_field, num_workers, cache_dir):

    def preprocess_function(examples):
        print(label_field)
        # create a result dictionary for each sample
        results = {key: [] for key in label_field + ['dialogue_a', 'dialogue_b']}
        
        # process the dialogue template
        for conversation, evaluations in zip(examples['conversation'], examples.get('evaluations', [])):
            try:
                # Build a complete multi-turn dialogue
                dialogue_a_messages = []
                dialogue_b_messages = []
                
                # Extract the dialogue content
                for i in range(0, len(conversation)-1, 2):
                    if i+1 < len(conversation):
                        # User message
                        if 'text' in conversation[i]:
                            human_query = conversation[i]['text']
                            dialogue_a_messages.append({'role': 'user', 'content': human_query})
                            dialogue_b_messages.append({'role': 'user', 'content': human_query})
                        # Assistant message
                        if 'A' in conversation[i+1] and 'B' in conversation[i+1]:
                            dialogue_a_messages.append({'role': 'assistant', 'content': conversation[i+1]['A']})
                            dialogue_b_messages.append({'role': 'assistant', 'content': conversation[i+1]['B']})
                
                # Apply the chat template
                results['dialogue_a'].append(tokenizer.apply_chat_template(dialogue_a_messages, tokenize=False))
                results['dialogue_b'].append(tokenizer.apply_chat_template(dialogue_b_messages, tokenize=False))
                
                # Initialize the label list for each dimension
                dimension_labels = {dim: [] for dim in label_field}
                
                # Process each annotator's evaluation
                for eval_data in evaluations:
                    if 'evaluation' in eval_data:
                        try:
                            # Parse the evaluation JSON
                            evaluation = json.loads(eval_data['evaluation'])
                            
                            # Collect the evaluation of each dimension
                            for dim in label_field:
                                if dim in evaluation:
                                    if evaluation[dim] == "A":
                                        dimension_labels[dim].append(0)  # A wins
                                    elif evaluation[dim] == "B":
                                        dimension_labels[dim].append(1)  # B wins
                                    else:  # Fair
                                        dimension_labels[dim].append(-1)  # Ignore the fair case
                        except Exception as e:
                            print(f"Error parsing evaluation: {e}")
                            continue
                
                # Add the label of each dimension to the result
                for dim in label_field:
                    if dimension_labels[dim]:
                        labels = torch.tensor(dimension_labels[dim], dtype=torch.long)
                        results[dim].append(labels)
                    else:
                        results[dim].append(torch.tensor([], dtype=torch.long))
            except Exception as e:
                print(f"Error processing conversation: {e}")
                for key in results:
                    if key == 'dialogue_a' or key == 'dialogue_b':
                        results[key].append("")
                    else:
                        results[key].append(torch.tensor([], dtype=torch.long))
        
        return results

    train_datasets = {}
    loaded_datasets = {}
    
    # Load the dataset
    for path in dataset_paths:
        try:
            dataset = datasets.load_dataset("json", data_files=path, cache_dir=cache_dir)
            print(f"loaded {path} {dataset}")
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            continue
        
        for split, data in dataset.items():
            try:
                dataset[split] = data.map(
                    preprocess_function,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=data.column_names,  
                    keep_in_memory=True,
                    desc="preprocessing new columns on dataset",
                )
            except Exception as e:
                print(f"Error preprocessing dataset {path}/{split}: {e}")
                continue
        
        if isinstance(dataset, datasets.DatasetDict):
            if "train" in dataset and len(dataset) == 1:
                loaded_datasets[path] = dataset["train"]
            else:
                for split, ds in dataset.items():
                    loaded_datasets[f"{path}/{split}"] = ds
        else:
            loaded_datasets[path] = dataset

    # Filter and split the dataset
    for path, dataset in loaded_datasets.items():
        try:
            try:
                filtered_dataset = dataset.filter(
                    LabelFilter(label_field), 
                    num_proc=num_workers, 
                    keep_in_memory=True
                )              
                print(f"Filtering dataset completed. Original dataset: {len(dataset)} samples, filtered: {len(filtered_dataset)} samples")
            except Exception as e:
                print(f"Error filtering dataset {path}: {e}. Using the original dataset.")
                filtered_dataset = dataset
            
            dataset_name = os.path.basename(path)
            
            if dataset_name in train_datasets:
                train_datasets[dataset_name] = datasets.concatenate_datasets([train_datasets[dataset_name], filtered_dataset])
            else:
                train_datasets[dataset_name] = filtered_dataset
                
        except Exception as e:
            print(f"Error processing dataset {path}: {e}")
            continue
    
    return train_datasets