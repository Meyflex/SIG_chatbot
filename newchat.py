import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments ,
from trl import SFTTrainer  # Import SFTTrainer for instruction-based fine-tuning
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Check if GPU is available and set the device accordingly
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)




if torch.cuda.device_count() > 1: # If more than 1 GPU
    print(torch.cuda.device_count())
    model.is_parallelizable = True
    model.model_parallel = True

# Read and prepare the data
data_path = 'niagara.data'  # Update with your file path
data = pd.read_csv(data_path, sep=';', header=None, names=['Code', 'Description', 'Lien', 'Detail', 'Flag'], on_bad_lines='skip')
data['input'] = "C'est quoi le code " + data['Code'] + " ?"
data['output'] = data['Description'] + ". " + data['Detail']

# Create a Hugging Face dataset
dataset = Dataset.from_pandas(data[['input', 'output']])

train_prompts, eval_prompts = train_test_split(tokenized_prompts, test_size=0.1, random_state=42)

def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "[INST]Use the provided input to create an instruction that could have been used to generate the response with an LLM."
  response = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  input = sample["response"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += system_message
  full_prompt += "\n" + input
  full_prompt += "[/INST]"
  full_prompt += response
  full_prompt += eos_token

  return full_prompt

# Training arguments
training_arg = TrainingArguments(
    output_dir = "./results_deep",
    #num_train_epochs=5,
    max_steps = 1000, # comment out this line if you want to train in epochs
    per_device_train_batch_size = 32,
    warmup_steps = 0.03,
    logging_steps=10,
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=10, # comment out this line if you want to evaluate at the end of each epoch
    learning_rate=2.5e-5,
    bf16=True,
)

# Trainer initialization
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=TrainingArguments,,     train_datasettrain_promptss,     eval_dataseteval_prompts,s
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./trained_model_deep")
tokenizer.save_pretrained("./trained_model_deep")

# Test the model with examples from the test set
for i in range(5):
    inputs = tokenizer(test_dataset[i]['input'], return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    print(f"Input: {test_dataset[i]['input']}")
    print(f"Expected Output: {test_dataset[i]['output']}")
    print(f"Model Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
