from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from functools import partial

def test_model(model, tokenizer, test_examples, device):
    model.eval()
    for example in test_examples:
        instruction = """
        Extract entities and relations from the following sentence.
        Example input: Emma borrowed a pen from Liam.
        Entities: [Emma, Liam, pen]; Relations: [(Emma, borrowed, pen), (pen, from, Liam)].
        Now do this for the following sentence: """
        input_text = instruction + example
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=200,
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
        f"""Extract entities and relations from the following sentence.
        Example input: Emma borrowed a pen from Liam.
        Entities: [Emma, Liam, pen]; Relations: [(Emma, borrowed, pen), (pen, from, Liam)].
        Now do this for the following sentence: {input_text}"""
        for input_text in examples["input_text"]
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define custom cache directory
    custom_cache_dir = "/dtu/blackhole/09/214057/llama"
    
    # Load tokenizer and model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=custom_cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    
    # Define training data
    data = {
        "input_text": [
            "Sophia handed the keys to Noah.",
            "Mason gave Olivia a bouquet of flowers.",
            "Ella sent a package to Lucas.",
            "Isabella returned a book to Ethan.",
            "James brought Ava a coffee.",
            "Mia wrote a letter to Benjamin.",
            "Charlotte gifted a painting to Henry.",
            "Amelia showed a map to Alexander.",
            "Logan explained the project to Harper.",
            "Lily delivered a message to Jack.",
            "Evelyn sold a car to Daniel.",
            "Grace taught a lesson to Samuel.",
            "Scarlett handed a folder to Leo.",
            "Zoey passed a note to Elijah.",
            "Chloe introduced William to Layla.",
            "Aria gave a speech to David.",
            "Hannah read a story to Gabriel.",
            "Violet showed a picture to Owen.",
            "Aurora prepared a meal for Michael."
        ],
        "target_text": [
            "Entities: [Sophia, Noah, keys]; Relations: [(Sophia, handed, keys), (keys, to, Noah)]",
            "Entities: [Mason, Olivia, bouquet]; Relations: [(Mason, gave, bouquet), (bouquet, to, Olivia)]",
            "Entities: [Ella, Lucas, package]; Relations: [(Ella, sent, package), (package, to, Lucas)]",
            "Entities: [Isabella, Ethan, book]; Relations: [(Isabella, returned, book), (book, to, Ethan)]",
            "Entities: [James, Ava, coffee]; Relations: [(James, brought, coffee), (coffee, to, Ava)]",
            "Entities: [Mia, Benjamin, letter]; Relations: [(Mia, wrote, letter), (letter, to, Benjamin)]",
            "Entities: [Charlotte, Henry, painting]; Relations: [(Charlotte, gifted, painting), (painting, to, Henry)]",
            "Entities: [Amelia, Alexander, map]; Relations: [(Amelia, showed, map), (map, to, Alexander)]",
            "Entities: [Logan, Harper, project]; Relations: [(Logan, explained, project), (project, to, Harper)]",
            "Entities: [Lily, Jack, message]; Relations: [(Lily, delivered, message), (message, to, Jack)]",
            "Entities: [Evelyn, Daniel, car]; Relations: [(Evelyn, sold, car), (car, to, Daniel)]",
            "Entities: [Grace, Samuel, lesson]; Relations: [(Grace, taught, lesson), (lesson, to, Samuel)]",
            "Entities: [Scarlett, Leo, folder]; Relations: [(Scarlett, handed, folder), (folder, to, Leo)]",
            "Entities: [Zoey, Elijah, note]; Relations: [(Zoey, passed, note), (note, to, Elijah)]",
            "Entities: [Chloe, William, Layla]; Relations: [(Chloe, introduced, William), (William, to, Layla)]",
            "Entities: [Aria, David, speech]; Relations: [(Aria, gave, speech), (speech, to, David)]",
            "Entities: [Hannah, Gabriel, story]; Relations: [(Hannah, read, story), (story, to, Gabriel)]",
            "Entities: [Violet, Owen, picture]; Relations: [(Violet, showed, picture), (picture, to, Owen)]",
            "Entities: [Aurora, Michael, meal]; Relations: [(Aurora, prepared, meal), (meal, for, Michael)]"
        ]
    }
    
    # Create dataset and tokenize
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer),  # Pass tokenizer to the function
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        learning_rate=2e-4,
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
    
    # Test before training
    print("\nTesting before training:")
    test_examples = [
        "Alice gave Bob a book.",
        "John sent an email to Mary."
    ]
    test_model(model, tokenizer, test_examples, device)
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Test after training
    print("\nTesting after training:")
    test_model(model, tokenizer, test_examples, device)

if __name__ == "__main__":
    main()