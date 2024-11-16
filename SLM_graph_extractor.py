from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from functools import partial
import json
import os

with open("instructions.txt") as f:
    instructions = f.read()

def test_model(model, tokenizer, test_examples, device):
    model.eval()
    for example in test_examples:
        input_text = instructions + "Input: " + example + "\nOutput: "
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=400,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Input: {example}")
            print(f"Generated Output: {output_text}")
            print("-" * 80)
            
        except RuntimeError as e:
            print(f"Error generating for input: {example}")
            print(f"Error message: {str(e)}")
            continue

def tokenize_function(examples, tokenizer):
    # Create prompt template for each example
    prompts = [
        instructions + "Input: " + example + "\nOutput: "
        # instructions + input_text
        for example in examples["input_text"]
    ]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )["input_ids"]
    
    # Replace padding tokens with -100 in labels
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs

def save_model(model, tokenizer, output_dir):
    """Save the model and tokenizer"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model(base_model_name, adapter_path, custom_cache_dir):
    """Load a fine-tuned model with its adapter weights"""
    print(f"Loading base model {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=custom_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=custom_cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter weights from {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="auto"
        )
    else:
        model = base_model
        
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    return model, tokenizer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths and configurations
    custom_cache_dir = "/dtu/blackhole/09/214057/llama"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = "./results"
    final_adapter_dir = os.path.join(output_dir, "final_lora_adapter")
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_name, None, custom_cache_dir)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    
    # Load and prepare training data
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        learning_rate=2e-5,
        warmup_steps=100,
        save_strategy="epoch",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        save_total_limit=1,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Test examples
    test_examples = [
        "Alice gave Bob a book.",
        "Violet showed Owen a picture of their family and an album filled with old memories.",
        "Aurora prepared a three-course meal for Michael, including a special dessert with candles."
    ]
    
    # Test before training
    print("\nTesting before training:")
    test_model(model, tokenizer, test_examples, device)
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Save the final model and adapter weights
    print("\nSaving final model...")
    save_model(trainer.model, tokenizer, final_adapter_dir)
    
    # Load the fine-tuned model for testing
    print("\nLoading fine-tuned model for testing...")
    fine_tuned_model, _ = load_model(model_name, final_adapter_dir, custom_cache_dir)
    
    # Test after training
    print("\nTesting after training:")
    test_model(fine_tuned_model, tokenizer, test_examples, device)

if __name__ == "__main__":
    main()
