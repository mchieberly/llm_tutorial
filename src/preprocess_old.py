# Preprocess

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

TEST_SPLIT = 0.2
BASE_MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

def formatting_func(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return text

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

def main():
	fsdp_plugin = FullyShardedDataParallelPlugin(
		state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
		optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
	)

	accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

	wandb.login()
	wandb_project = "llm_tutorial"
	if len(wandb_project) > 0:
		os.environ["WANDB_PROJECT"] = wandb_project

	dataset = load_dataset(
		'csv',
		data_files=['data/allergies.csv', 
					'data/conditions.csv', 
					'data/encounters.csv',
					'data/immunizations.csv',
					'data/observations.csv',
					'data/patients.csv',
					'data/payer_transitions.csv',
					'data/providers.csv',
					'data/careplans.csv',
					'data/devices.csv',
					'data/imaging_studies.csv',
					'data/medications.csv',
					'data/organizations.csv',
					'data/payers.csv',
					'data/procedures.csv',
					'data/supplies.csv',
					],
		split='train'
	)

	train_test_split = dataset.train_test_split(test_size=TEST_SPLIT)
	train_dataset = train_test_split['train']
	test_dataset = train_test_split['test']

	base_model_id = BASE_MODEL_ID
	model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16, load_in_8bit=True)


	tokenizer = AutoTokenizer.from_pretrained(
		base_model_id,
		padding_side="left",
		add_eos_token=True,
		add_bos_token=True,
		use_fast=False,
	)

	def generate_and_tokenize_prompt(prompt):
		return tokenizer(formatting_func(prompt))

	tokenizer.pad_token = tokenizer.eos_token
	tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
	tokenized_val_dataset = test_dataset.map(generate_and_tokenize_prompt)

	plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

if __name__ == '__main__':
	main()
