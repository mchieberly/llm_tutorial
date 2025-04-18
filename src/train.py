from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import transformers
import torch
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

BASE_MODEL = "microsoft/phi-2"
MAX_LENGTH = 128
QLORA_R = 4
QLORA_ALPHA = 8
QLORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
TRAIN_EPOCHS = 1
MAX_STEPS = 100
LEARNING_RATE = 1e-4
LOGGING_STEPS = 20

def formatting_func(example):
    text = (
        f"### Instruction: Analyze the patient case and diagnose COVID-19.\n"
        f"### Patient Case:\n{example['text']}\n"
        f"### Diagnosis: {'COVID-19' if example['label'] == 'COVID-19' else 'Other'}"
    )
    return {"text": text}

def main():
	# Load and prepare dataset
	dataset = load_dataset(
		"csv",
		data_files={
			"train": "processed_data/train.csv",
			"validation": "processed_data/validation.csv",
			"test": "processed_data/test.csv"
		}
	).remove_columns(["PATIENT", "split"])

	# Apply formatting
	dataset = dataset.map(formatting_func)

	# 4-bit quantization config
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16
	)

	# Load model
	model = transformers.AutoModelForCausalLM.from_pretrained(
		BASE_MODEL,
		quantization_config=bnb_config,
		device_map="auto",
		trust_remote_code=True,
		attn_implementation="flash_attention_2"
	)

	tokenizer = transformers.AutoTokenizer.from_pretrained(
		BASE_MODEL,
		padding_side="left",
		add_eos_token=True,
		add_bos_token=True,
		use_fast=False,
	)
	tokenizer.pad_token = tokenizer.eos_token

	# Tokenization function
	def tokenize_function(examples):
		return tokenizer(
			[text + tokenizer.eos_token for text in examples["text"]],
			truncation=True,
			max_length=MAX_LENGTH,
			padding="max_length",
			return_tensors=None
		)

	# Process dataset
	tokenized_dataset = dataset.map(
		tokenize_function,
		batched=True,
		remove_columns=["text", "label"]
	)

	# QLoRA config
	peft_config = LoraConfig(
		r=QLORA_R,
		lora_alpha=QLORA_ALPHA,
		target_modules=[
				"Wqkv",
				"fc1",
				"fc2",
			],
		lora_dropout=QLORA_DROPOUT,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, peft_config)

	# Training arguments
	training_args = transformers.TrainingArguments(
		output_dir="./covid_classifier",
		per_device_train_batch_size=BATCH_SIZE,
		per_device_eval_batch_size=BATCH_SIZE,
		gradient_accumulation_steps=GRAD_ACCUM_STEPS,
		max_steps=MAX_STEPS,
		eval_strategy="epoch",
		learning_rate=LEARNING_RATE,
		logging_steps=LOGGING_STEPS,
		fp16=True,
		optim="adamw_torch_4bit",
		report_to="none",
		save_strategy="epoch",
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss"
	)

	# Data collator
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False
	)

	# Initialize Trainer
	trainer = transformers.Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset["train"],
		eval_dataset=tokenized_dataset["validation"],
		data_collator=data_collator,
	)

	# Start training
	model.config.use_cache = False
	trainer.train()

	test_results = trainer.evaluate(tokenized_dataset["test"])
	print("Test set results:", test_results)

if __name__ == "__main__":
	main()
