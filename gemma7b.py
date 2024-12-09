import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import transformers
from trl import SFTTrainer, SFTConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_cuda():
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

def load_model_and_tokenizer(model_id, cache_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Set padding side to right
    return model, tokenizer

def load_and_clean_dataset(dataset_name, cache_dir):
    dataset = load_dataset(dataset_name, "en", cache_dir=cache_dir, trust_remote_code=True)
    def remove_extra_columns(example):
        example["entities"] = [entity["surfaceform"] for entity in example["entities"]]
        example["relations"] = [
            {
                "subject": example["entities"][relation["subject"]],
                "predicate": relation["predicate"],
                "object": example["entities"][relation["object"]],
            }
            for relation in example["relations"]
        ]
        return example
    return dataset.map(remove_extra_columns)

def preprocess_function(data_point):
    """
    Convert entities and relations to the expected output text format.
    """
    query = data_point['text']
    entities = ", ".join([f'"{entity}"' for entity in data_point['entities']])
    relations = "\n".join([f'"{relations}"' for relations in data_point['relations']])

    text = f"Given the following text, identify and extract all entities and their relations. Query: {query}\n Entities: [{entities}]\nRelations:\n{relations}"
    return text

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    check_cuda()
    
    model_id = "google/gemma-7b-it"
    cache_dir = "SLM/gemma7b"  # Specify an alternative cache directory
    dataset_name = "Babelscape/SREDFM"
    
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)
    dataset = load_and_clean_dataset(dataset_name, cache_dir)
    
    tokenized_datasets = dataset.map(lambda dp: {"model_input": preprocess_function(dp)})
    logger.info(tokenized_datasets["test"][0])
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    modules = find_all_linear_names(model)
    logger.info(modules)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")
    
    torch.cuda.empty_cache()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        dataset_text_field='model_input',
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,  # Increase batch size if memory allows
            gradient_accumulation_steps=8,
            warmup_steps=100,  # Add warmup steps
            max_steps=1000,  # Increase max steps for better training
            learning_rate=2e-4,
            logging_steps=10,  # Log less frequently to reduce overhead
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            fp16=True,  # Use mixed precision training
            dataloader_num_workers=4,  # Optimize data loading
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False
    trainer.train()
    
    new_model = "gemma7b-trained"
    trainer.model.save_pretrained(new_model)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        cache_dir="SLM/gemma7b_trained"  # Specify an alternative cache directory
    )
    merged_model = PeftModel.from_pretrained(base_model, new_model)
    merged_model = merged_model.merge_and_unload()
    
    merged_model.to("cpu")
    tokenizer.save_pretrained("gemma7b_trained")
    merged_model.save_pretrained("gemma7b_trained", safe_serialization=True)
    
    merged_model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)

if __name__ == "__main__":
    main()
