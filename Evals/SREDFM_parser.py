from datasets import load_dataset

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

if __name__ == "__main__":
    # Load only the test split
    limit = 1000
    test_dataset = load_dataset("Babelscape/SREDFM", "en", split="test", trust_remote_code=True)
    entities = test_dataset["entities"][:limit]
    relations = test_dataset["relations"][:limit]
    texts = test_dataset["text"][:limit]

    cleaned_dataset = test_dataset.map(remove_extra_columns)
    print(cleaned_dataset[0])
