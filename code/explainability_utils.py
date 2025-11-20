import random
import math
from typing import List, Dict, Any, Tuple


def shapley_vote_contrib(per_model_probs, weights, target_class, nsamples=256):
    """
    Per-signal Shapley-style contribution of each model to ensemble prob(target_class).
    
    Args:
        per_model_probs: [M][C] after temperature scaling
        weights: [M] voting weights
        target_class: class index for which to compute contributions
        nsamples: number of Monte Carlo samples
    
    Returns:
        List[float]: contribution of each model to final probability
    """
    M = len(per_model_probs)
    contrib = [0.0] * M
    
    for _ in range(nsamples):
        perm = list(range(M))
        random.shuffle(perm)
        agg = [0.0] * len(per_model_probs[0])
        
        def norm(v):
            s = sum(v)
            return [x / s if s > 0 else 0.0 for x in v]
        
        for idx in perm:
            before = norm(agg)[target_class] if sum(agg) > 0 else 0.0
            for c in range(len(agg)):
                agg[c] += weights[idx] * per_model_probs[idx][c]
            after = norm(agg)[target_class]
            contrib[idx] += (after - before)
    
    return [v / float(nsamples) for v in contrib]


def compute_explanation_metrics(per_model_probs, weights, temperatures, aggregate_probs):
    """
    Compute comprehensive explanation metrics for an ensemble decision.
    
    Args:
        per_model_probs: [M][C] model probabilities
        weights: [M] model weights  
        temperatures: [M] temperature scaling factors
        aggregate_probs: [C] final ensemble probabilities
        
    Returns:
        Dict containing various explanation metrics
    """
    num_models = len(per_model_probs)
    num_classes = len(per_model_probs[0])
    top_class = max(range(num_classes), key=lambda i: aggregate_probs[i])
    
    # Shapley contributions for top prediction
    shapley_contrib = shapley_vote_contrib(per_model_probs, weights, top_class)
    
    # Disagreement metrics
    disagreements = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            model_i_top = max(range(num_classes), key=lambda c: per_model_probs[i][c])
            model_j_top = max(range(num_classes), key=lambda c: per_model_probs[j][c])
            if model_i_top != model_j_top:
                disagreements.append((i, j, model_i_top, model_j_top))
    
    # Confidence spread
    confidences = [max(probs) for probs in per_model_probs]
    conf_std = math.sqrt(sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) / len(confidences))
    
    # Entropy analysis
    entropies = []
    for probs in per_model_probs:
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in probs)
        entropies.append(entropy)
    
    return {
        "shapley_contributions": shapley_contrib,
        "disagreement_count": len(disagreements),
        "disagreement_pairs": disagreements,
        "confidence_spread": conf_std,
        "mean_confidence": sum(confidences) / len(confidences),
        "model_entropies": entropies,
        "mean_entropy": sum(entropies) / len(entropies),
        "weight_entropy": -sum(w * math.log(max(w, 1e-9)) for w in weights),
        "effective_models": sum(1 for w in weights if w > 0.01)  # models with > 1% weight
    }


def analyze_model_agreement(per_model_probs, threshold=0.7):
    """
    Analyze agreement patterns between models.
    
    Args:
        per_model_probs: [M][C] model probabilities
        threshold: confidence threshold for considering a "strong" prediction
        
    Returns:
        Dict with agreement analysis
    """
    num_models = len(per_model_probs)
    num_classes = len(per_model_probs[0])
    
    # Find strong predictions (confidence > threshold)
    strong_predictions = []
    for i, probs in enumerate(per_model_probs):
        max_prob = max(probs)
        if max_prob >= threshold:
            pred_class = max(range(num_classes), key=lambda c: probs[c])
            strong_predictions.append((i, pred_class, max_prob))
    
    # Agreement matrix
    agreement_matrix = {}
    for i in range(num_models):
        for j in range(i + 1, num_models):
            pred_i = max(range(num_classes), key=lambda c: per_model_probs[i][c])
            pred_j = max(range(num_classes), key=lambda c: per_model_probs[j][c])
            agreement_matrix[(i, j)] = 1.0 if pred_i == pred_j else 0.0
    
    # Class-wise support
    class_support = [0] * num_classes
    for probs in per_model_probs:
        pred_class = max(range(num_classes), key=lambda c: probs[c])
        class_support[pred_class] += 1
    
    return {
        "strong_predictions": strong_predictions,
        "agreement_matrix": agreement_matrix,
        "class_support": class_support,
        "consensus_strength": max(class_support) / num_models,
        "num_strong_predictions": len(strong_predictions)
    }


def generate_text_explanation(explanation_metrics, model_names=None):
    """
    Generate human-readable text explanation of ensemble decision.
    
    Args:
        explanation_metrics: output from compute_explanation_metrics
        model_names: optional list of model names
        
    Returns:
        str: text explanation
    """
    if model_names is None:
        model_names = [f"Model {i}" for i in range(len(explanation_metrics["shapley_contributions"]))]
    
    contrib = explanation_metrics["shapley_contributions"]
    top_contributor = max(range(len(contrib)), key=lambda i: contrib[i])
    
    text = f"Ensemble decision driven primarily by {model_names[top_contributor]} "
    text += f"(Δp = {contrib[top_contributor]:.3f}). "
    
    if explanation_metrics["disagreement_count"] > 0:
        text += f"Models showed {explanation_metrics['disagreement_count']} disagreements. "
    else:
        text += "All models agreed on the prediction. "
    
    text += f"Confidence spread: {explanation_metrics['confidence_spread']:.3f}, "
    text += f"effective models: {explanation_metrics['effective_models']}."
    
    return text