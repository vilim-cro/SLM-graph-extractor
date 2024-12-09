import re

replacements = [
    ("'Non", "Non"),
    ("[\"\"The First Celestial", "\"\"The First Celestial"),
    ("[\"\"The Silent Serin", "\"\"The Silent Serin"),
    ("[\"\"The Untuned Ventriloquist", "\"\"The Untuned Ventriloquist"),
    ("[\"\"The Cannibal Manifesto in the Dark", "\"\"The Cannibal Manifesto in the Dark")
]

def parse_gemma_output(file_path: str) -> dict:
    with open(file_path, encoding='utf-8') as f:
        data = f.read()

    # Clean ' from the data
    data = re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9\s-])", r"\1\2", data)
    for replacement in replacements:
        data = data.replace(replacement[0], replacement[1])
    sections = data.split('[')
    all_entities = []
    all_relations = []
    skip = 0
    first_section = True
    for s in sections[1:]:
        if s[:1] == "{" or s[:1].isalnum() or s[:1] in ".=":
            if s[:1] == "{":
                skip += 1
            if not s[:1].isalnum() and not s[:1] in ".=" and s[:1] != "{":
                print(s)
            continue

        # Handle cases where model didnt generate anything
        if skip % 2 != 0:
            raise ValueError("Unbalanced brackets")
        count = skip // 2 - 1
        if first_section:
            count += 1
            first_section = False
        for _ in range(count):
            all_entities.append([])
            all_relations.append([])
        skip = 0

        end_index = s.find(']')
        if end_index != -1:
            entities = s[:end_index].replace("\"\"", "").split(', ')
            relations_as_list = s[end_index + 4:].replace("\"\"", "").split('\n')
            if "}" not in relations_as_list[-1]:
                relations_as_list = relations_as_list[:-1]
            else:
                relations_as_list[-1] = relations_as_list[-1][:relations_as_list[-1].find("}") + 1]
            relations = list(set(relations_as_list))
            try:
                relations = [eval(relation) for relation in relations if relation]
            except SyntaxError as _:
                # print(f"Error parsing the following relations {relations}")
                all_entities.append([])
                all_relations.append([])
                continue

            all_entities.append(entities)
            all_relations.append(relations)
    return {"entities": all_entities, "relations": all_relations}

def parse_rebel_output(file_path: str) -> dict:
    with open(file_path, encoding='utf-8') as f:
        data = f.read()

    sections = data.split('\n#\n')
    all_entities = []
    all_relations = []
    for s in sections[:-1]:
        s = s.strip()
        entities, relations = s.split('\n')
        all_entities.append(eval(entities))
        all_relations.append(eval(relations))

    return {"entities": all_entities, "relations": all_relations}

# parse_gemma_output("C:\\Users\\vbran\\Desktop\\Programi\\Python\\SLM-graph-extractor\\Outputs\\gemma7b_entities_relations.csv")
