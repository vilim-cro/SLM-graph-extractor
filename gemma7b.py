# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import transformers
from trl import SFTConfig, SFTTrainer


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Load Mistral 7B model and tokenizer
model_id = "google/gemma-7b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, cache_dir="SLM/gemma7b")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, cache_dir="SLM/gemma7b")

dataset = load_dataset("Babelscape/SREDFM", "en", cache_dir="SLM/datasets", trust_remote_code=True)

def remove_extra_columns(example):
    example["entities"] = [
        entity["surfaceform"]
        for entity in example["entities"]
    ]
    example["relations"] = [
        {
            "subject": example["entities"][relation["subject"]],
            "predicate": relation["predicate"],
            "object": example["entities"][relation["object"]],
        }
        for relation in example["relations"]
    ]
    return example

cleaned_dataset = dataset.map(remove_extra_columns)
print(cleaned_dataset["test"][0])

def generate_prompt(data_point):
    """
    Generate input text based on a prompt, task instruction, context info, and answer.

    :param data_point: dict: Data point from the Babelscape SREDFM dataset
    :return: str: Formatted prompt text
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                  'appropriately completes the request.\n\n'

    # Task instruction asking to extract entities and relations
    task_instruction = (
        "Given the following text, identify and extract all entities and their relations. "
        "Entities could be people, organizations, locations, etc., .I want you to foucs on surfaceform only in the entities."
        "and relations could be actions, associations, etc"
    )
    
    # Formatted prompt incorporating the data point text
    text = (
        f"{prefix_text} {task_instruction}\n\n"
        f"Text: {data_point['text']}\n\n"
        "Entities and Relations:\n"
        f"Entities: {data_point['entities']}\n"
        f"Relations: {data_point['relations']}\n"
    )

    return text

text_column = [generate_prompt(data_point) for data_point in cleaned_dataset['train']]
cleaned_dataset['train'] = cleaned_dataset['train'].add_column("prompt", text_column)
train_data = cleaned_dataset['train'].to_pandas()

text_column = [generate_prompt(data_point) for data_point in cleaned_dataset['test']]
cleaned_dataset['test'] = cleaned_dataset['test'].add_column("prompt", text_column)
test_data = cleaned_dataset['test'].to_pandas()

print(train_data['prompt'])

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# print(model)

def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)
print(modules)

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
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

# Create SFTConfig
trainer = SFTTrainer(
    model=model,
    train_dataset=cleaned_dataset['train'],
    eval_dataset=cleaned_dataset['test'],
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

new_model = "mistral-Code-Instruct-Finetune-test"
trainer.model.save_pretrained(new_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    cache_dir = "SLM/gemma7b_finetuned"
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.to("cpu")
tokenizer.save_pretrained("gemma7b_trained")
merged_model.save_pretrained("gemma7b_trained",safe_serialization=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

merged_model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

#####################################################################################################
# def get_completion(query: str, model, tokenizer) -> str:
#     # Define the prompt template
#     prompt_template = """
#     <start_of_turn>user
#     Generate entities and relations for the query.
#     {query}
#     <end_of_turn>\n<start_of_turn>model
#     """
#     # Format the prompt
#     prompt = prompt_template.format(query=query)

#     # Tokenize the input
#     encodeds = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

#     # Move input tensors to the same device as the model
#     encodeds = {key: value.to("cpu") for key, value in encodeds.items()}

#     # Generate predictions
#     generated_ids = model.generate(
#         **encodeds, 
#         max_new_tokens=1000, 
#         do_sample=True, 
#         pad_token_id=tokenizer.eos_token_id
#     )

#     # Decode the output
#     decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     return decoded

# result = get_completion(query="Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.", model=merged_model, tokenizer=tokenizer)
# print(result)
