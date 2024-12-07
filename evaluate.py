from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_graph_metrics(pred_nodes, true_nodes, pred_relations, true_relations):
    # Compute for nodes
    node_precision = len(set(pred_nodes) & set(true_nodes)) / len(set(pred_nodes))
    node_recall = len(set(pred_nodes) & set(true_nodes)) / len(set(true_nodes))
    node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall)

    # Compute for relationships
    relation_precision = len(set(pred_relations) & set(true_relations)) / len(set(pred_relations))
    relation_recall = len(set(pred_relations) & set(true_relations)) / len(set(true_relations))
    relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall) if\
            (relation_precision + relation_recall) > 0 else 0

    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "relation_f1": relation_f1,
    }

def evaluate(true_nodes: list, predicted_nodes: list, true_relations: dict, predicted_relations: dict) -> dict:
    """Takes in the true and predicted nodes and relations and returns the node and relation scores
    true_nodes: list of true nodes represented as strings
    predicted_nodes: list of predicted nodes represented as strings
    true_relations: dictionary of true relations in format {'subject': str, 'predicate': str, 'object': str}
    predicted_relations: dictionary of predicted relations in format {'subject': str, 'predicate': str, 'object': str}
    """
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

    aligned_similarity = cosine_similarity(true_embeddings, pred_embeddings)
    aligned_similarity = [max(row) for row in aligned_similarity]

    aligned_similarity_mean = sum(aligned_similarity) / len(aligned_similarity)

    metrics["aligned_similarity"] = aligned_similarity_mean

    f1_weight = 0.3
    relations_score = metrics["relation_f1"] * f1_weight +\
        metrics["aligned_similarity"] * (1-f1_weight)
    return {'node_score': metrics['node_f1'], 'relations_score': relations_score}


# true_nodes = ["Ema", "book", "Thomas", "library"]
# predicted_nodes = ["Ema", "book", "Thomas"]

# true_relations = [("Ema", "gave", "book"), ("book", "to", "Thomas"),
#                   ("Ema", "gave", "book"), ("book", "to", "Thomas")]
# predicted_relations = [("Ema", "gave", "book"), ("book", "to", "Thomas"),
#                        ("Ema", "gave", "the book"), ("book", "to", "Thomas")]

# evaluate(true_nodes, pred_nodes, true_relations, pred_relations)
