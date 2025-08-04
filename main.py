from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,                  # Rank (lower = less VRAM, higher = better quality)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Target Llama's attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ~0.1% of parameters trained

from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.json")
def format_instruction(sample):
    return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"

dataset = dataset.map(lambda x: {"text": format_instruction(x)})

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./llama2-7b-finetuned",
    per_device_train_batch_size=4,     # Reduce if OOM (1-4)
    gradient_accumulation_steps=4,      # Compensate for small batch size
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    optim="paged_adamw_8bit",          # Memory-efficient optimizer
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",                 # Log to Weights & Biases (optional)
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model, 
    args=training_args,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=1024,               # Reduce to 512 if OOM
    tokenizer=tokenizer,
    packing=True,                      # Pack sequences efficiently
)

trainer.train()

model.save_pretrained("llama2-7b-lora-adapters")  # Save only adapters (small)
tokenizer.save_pretrained("llama2-7b-lora-adapters")

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "llama2-7b-lora-adapters")

inputs = tokenizer("### Instruction:\nExplain quantum entanglement\n\n### Response:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))