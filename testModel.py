import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer using the same one you used for fine-tuning
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Assuming you have the classic model loaded as `classic_model`
# and the fine-tuned model loaded as `fine_tuned_model`
classic_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./trained_model_deep")  # Adjust path as needed

# Make sure both models are in evaluation mode
classic_model.eval()
fine_tuned_model.eval()

# Move models to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classic_model.to(device)
fine_tuned_model.to(device)

def interactive_model_comparison():
    # Ensure pad_token is set for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    while True:
        input_text = input("Enter your question (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break

        encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        # Assuming classic_model and fine_tuned_model are already loaded and set to the correct device
        # Generate response with the classic model
        classic_output = classic_model.generate(input_ids, attention_mask=attention_mask, max_length=256)
        classic_response = tokenizer.decode(classic_output[0], skip_special_tokens=True)
        
        # Generate response with the fine-tuned model
        fine_tuned_output = fine_tuned_model.generate(input_ids, attention_mask=attention_mask, max_length=256)
        fine_tuned_response = tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)
        
        print(f"Input: {input_text}")
        print(f"Classic Model Output: {classic_response}")
        print(f"Fine-Tuned Model Output: {fine_tuned_response}\n")

# Make sure to define classic_model, fine_tuned_model, tokenizer, and device before calling this function
interactive_model_comparison()

