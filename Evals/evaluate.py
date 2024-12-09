from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

from output_parser import parse_gemma_output, parse_rebel_output
from SREDFM_parser import remove_extra_columns

def compute_graph_metrics(pred_nodes, true_nodes, pred_relations, true_relations):
    # Compute for nodes
    if len(pred_nodes) == 0:
        node_f1 = 0
    else:
        node_precision = len(set(pred_nodes) & set(true_nodes)) / len(set(pred_nodes))
        node_recall = len(set(pred_nodes) & set(true_nodes)) / len(set(true_nodes))
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall) if\
                (node_precision + node_recall) > 0 else 0

    # Compute for relationships
    if len(pred_relations) == 0:
        relation_f1 = 0
    else:
        relation_precision = len(set(pred_relations) & set(true_relations)) / len(set(pred_relations))
        relation_recall = len(set(pred_relations) & set(true_relations)) / len(set(true_relations))
        relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall) if\
                (relation_precision + relation_recall) > 0 else 0

    return {
        # "node_precision": node_precision,
        # "node_recall": node_recall,
        "node_f1": node_f1,
        # "relation_precision": relation_precision,
        # "relation_recall": relation_recall,
        "relation_f1": relation_f1,
    }

def evaluate(true_nodes: list, predicted_nodes: list, true_relations: dict, predicted_relations: dict) -> dict:
    """Takes in the true and predicted nodes and relations and returns the node and relation scores
    true_nodes: list of true nodes represented as strings
    predicted_nodes: list of predicted nodes represented as strings
    true_relations: dictionary of true relations in format {'subject': str, 'predicate': str, 'object': str}
    predicted_relations: dictionary of predicted relations in format {'subject': str, 'predicate': str, 'object': str}
    """

    if not predicted_nodes and not predicted_relations:
        return {'node_score': 0, 'relations_score': 0}

    true_relations_tuples = [(rel['subject'], rel['predicate'], rel['object'])
                             for rel in true_relations]
    predicted_relations_tuples = [(rel['subject'], rel['predicate'], rel['object'])
                                  for rel in predicted_relations]

    true_relations_text = [f"{rel['subject']} -> {rel['predicate']} -> {rel['object']}"
                            for rel in true_relations]
    predicted_relations_text = [f"{rel['subject']} -> {rel['predicate']} -> {rel['object']}"
                                for rel in predicted_relations]

    metrics = compute_graph_metrics(predicted_nodes, true_nodes,
                                    predicted_relations_tuples, true_relations_tuples)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    true_embeddings = model.encode(true_relations_text)
    pred_embeddings = model.encode(predicted_relations_text)

    if len(pred_embeddings) > 0:
        aligned_similarity = cosine_similarity(true_embeddings, pred_embeddings)
        aligned_similarity = [max(row) for row in aligned_similarity]

        aligned_similarity_mean = sum(aligned_similarity) / len(aligned_similarity)
    else:
        aligned_similarity_mean = 0

    metrics["aligned_similarity"] = aligned_similarity_mean

    f1_weight = 0.3
    relations_score = metrics["relation_f1"] * f1_weight +\
        metrics["aligned_similarity"] * (1-f1_weight)
    return {'node_score': metrics['node_f1'], 'relations_score': relations_score}

if __name__ == "__main__":
    start = 0
    end = 1000
    test_dataset = load_dataset("Babelscape/SREDFM", "en", split="test", trust_remote_code=True)
    test_dataset = test_dataset.map(remove_extra_columns)
    entities = test_dataset["entities"][start:end]
    relations = test_dataset["relations"][start:end]

    to_parse = [
        ("../Outputs/gemma2b_entities_relations.csv", parse_gemma_output, "Gemma 2b"),
        ("../Outputs/gemma7b_entities_relations.csv", parse_gemma_output, "Gemma 7b"),
        ("../Outputs/rebel_entities_relations.csv", parse_rebel_output, "Rebel"),
    ]

    for file_path, parser, name in to_parse:
        node_scores = []
        relations_scores = []
        results = parser(file_path)
        # print(len(entities), len(relations), len(results['entities']), len(results['relations']))
        for true_entities, true_relations, pred_entities, pred_relations in zip(
            entities, relations, results['entities'][start:end], results['relations'][start:end]
            ):
            scores = evaluate(true_entities, pred_entities, true_relations, pred_relations)
            node_scores.append(scores['node_score'])
            relations_scores.append(scores['relations_score'])
        # print(node_scores, relations_scores)
        print(f"{name} node score: {sum(node_scores) / len(node_scores)}")
        print(f"{name} relations score: {sum(relations_scores) / len(relations_scores)}")
        print(f"{name} node score without missing values: {sum(node_scores) / len([score for score in node_scores if score > 0])}")
        print(f"{name} relations score without missing values: {sum(relations_scores) / len([score for score in relations_scores if score > 0])}")
        print()
