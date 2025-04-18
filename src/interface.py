import gradio as gr
from peft import PeftModel
import torch
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	TextIteratorStreamer,
	pipeline,
	BitsAndBytesConfig
)
from threading import Thread

TIMEOUT = 300.0
MAX_NEW_TOKENS = 128
BASE_MODEL = "microsoft/phi-2"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
	BASE_MODEL,
	device_map="auto",
	trust_remote_code=True,
    quantization_config=quantization_config,
	torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_bos_token=True, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

phi2 = pipeline(
	"text-generation",
	tokenizer=tokenizer,
	model=base_model,
	pad_token_id=tokenizer.eos_token_id,
	eos_token_id=tokenizer.eos_token_id,
	device_map="auto"
)
phi2.model = PeftModel.from_pretrained(base_model, "./covid_classifier/checkpoint-100")

def generate(message, chat_history):
	"""Accepts a prompt and generates text using the phi-2 pipeline"""
	max_new_tokens = MAX_NEW_TOKENS
	instruction = "You are a doctor that answers questions from 'User' indicating whether or not a described patient has COVID-19." + \
		"You make your prediction based on the symptoms, medications, and other details that 'User' describes."
	final_prompt = f"Instruction: {instruction}\n"

	for sent, received in chat_history:
		final_prompt += "User: " + sent + "\n"
		final_prompt += "Assistant: " + received + "\n"

	final_prompt += "User: " + message + "\n"
	final_prompt += "Output:"

	if (len(tokenizer.tokenize(final_prompt)) >= tokenizer.model_max_length - max_new_tokens):
		final_prompt = "Instruction: Say 'Input exceeded context size, please clear the chat history and retry!' Output:"

	streamer = TextIteratorStreamer(
		tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=TIMEOUT
	)
	thread = Thread(
		target=phi2,
		kwargs={
			"text_inputs": final_prompt,
			"max_new_tokens": max_new_tokens,
			"streamer": streamer,
		},
	)
	thread.start()

	generated_text = ""
	for word in streamer:
		generated_text += word
		response = generated_text.strip()

		if "User:" in response:
			response = response.split("User:")[0].strip()

		if "Assistant:" in response:
			response = response.split("Assistant:")[1].strip()

		yield response


with gr.Blocks() as demo:
	gr.Markdown(
		"""
		# COVID Classifier Chatbot Using Phi-2 and QLoRA
		This chatbot was created using the Phi-2 2.7B model (https://huggingface.co/microsoft/phi-2). It has been tuned with QLoRA to answer questions about if a patient has COVID-19.
		
		Created by Malachi Eberly
		"""
	)

	chatbot = gr.ChatInterface(
		fn=generate,
		stop_btn=None,
		examples=[["A patient has nasal congestion, a headache, and a high heart rate. They also take a Hydrochlorothiazide 25 MG Oral Tablet. Indicate if this is COVID-19 or not."]]
	)

demo.queue().launch()
