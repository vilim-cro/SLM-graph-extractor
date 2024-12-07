from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from parser import remove_extra_columns
from evaluate import evaluate

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

limit = 10
test_dataset = load_dataset("Babelscape/SREDFM", "en", split="test", trust_remote_code=True)
test_dataset = test_dataset.map(remove_extra_columns)
entities = test_dataset["entities"][:limit]
relations = test_dataset["relations"][:limit]
texts = test_dataset["text"][:limit]

node_scores = []
relation_scores = []
for text, true_entity, true_relation in zip(texts, entities, relations):
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, 
                             return_tensors = 'pt')
    generated_tokens = model.generate(model_inputs["input_ids"].to(model.device),
                                    attention_mask=model_inputs["attention_mask"].
                                    to(model.device), **gen_kwargs)
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    for idx, sentence in enumerate(decoded_preds):
        predicted_relations = extract_triplets(sentence)
        predicted_entities = set()
        for relation in predicted_relations:
            predicted_entities.add(relation['subject'])
            predicted_entities.add(relation['object'])
        # print()
        # print()
        # print(f"Prediction entities: {predicted_entities}")
        # print(f"True entities: {true_entity}")
        # print()
        # print(f'Prediction triplets sentence {idx}')
        # print(predicted_relations)
        # print(f"True relations: {true_relation}")
        scores = evaluate(true_entity, predicted_entities,
                 true_relation, predicted_relations)
        node_scores.append(scores['node_score'])
        relation_scores.append(scores['relations_score'])

print(f"Node F1: {sum(node_scores) / len(node_scores)}")
print(f"Relation F1: {sum(relation_scores) / len(relation_scores)}")
