"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
import re
def clean_text(text: str) -> str:

  return re.sub(r"[^a-z]", " ", text.lower()).strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    return [tokenize(clean_text(text)) for text in texts]

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    
    np.random.seed(seed) 

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test

    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

# Step 5 - count_word_frequencies
from collections import Counter
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    counter = Counter()
    for doc in tokenized_docs:
        counter.update(doc)

    return dict(counter)

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # TODO: Keep the top max_size most frequent words; map each to an index in [0, V).
    if max_size <= 0:
        return {}

    sorted_words = sorted(
        word_counts.items(), key=lambda items: (-items[1], items[0])
    )[:max_size]

    return {
        word: idx for idx, (word, _) in enumerate(sorted_words)
    }

# Step 7 - tokens_to_bow
import numpy as np
from collections import Counter

def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    """Convert one document's token list into a bag-of-words count vector.

    Args:
        tokens: List of word tokens in the document.
        vocab: Dict mapping each kept word to its unique integer index [0, V).

    Returns:
        1D numpy array of shape (len(vocab),) containing occurrence counts.
    """
    bow = np.zeros(len(vocab), dtype=float)
    counter_token = Counter(tokens)

    for token, count in counter_token.items():
        if token in vocab:
            bow[vocab[token]] = count

    return bow

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    if not tokenized_docs:
        return np.zeros((0, len(vocab)), dtype=float)
    return np.array(
        [tokens_to_bow(doc, vocab) for doc in tokenized_docs], dtype=float
    )

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    return np.sum(bow_matrix > 0, axis=0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    return np.log((n_docs + 1) / (df + 1)) + 1

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]
    idf = compute_idf(df, n_docs)

    return idf

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    return np.where(z > 0, 1 / (1 + np.exp(-z)), np.exp(z) / (np.exp(z) + 1))

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    return sigmoid(X @ w + b)

# Step 15 - binary_cross_entropy
import numpy as np


def binary_cross_entropy(
    y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = np.asarray(w, dtype=float)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    bce_loss = -np.mean(
        y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)
    )

    l2_loss = 0.5 * l2_lambda * np.sum(w**2)

    return float(bce_loss + l2_loss)

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    n = X.shape[0]
    error = y_prob - y_true  # shape: (N,)

    dw = (X.T @ error) / n + l2_lambda * w

    db = float(np.mean(error))

    return dw, db

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    return np.zeros(n_features), 0.0

# Step 18 - gradient_descent_step
import numpy as np

def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lr: float,
    l2_lambda: float,
) -> tuple[np.ndarray, float, float]:
    """Run one full-batch gradient descent update for logistic regression with L2 regularization.

    Args:
        X: Feature matrix of shape (N, D).
        y: Binary labels of shape (N,).
        w: Weight vector of shape (D,).
        b: Bias scalar.
        lr: Learning rate.
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple of (w_new, b_new, loss) where:
            - w_new: Updated weights of shape (D,).
            - b_new: Updated bias as a float.
            - loss: Current step's BCE loss + 0.5 * L2 penalty before update.
    """
    n_samples = X.shape[0]

    y_prob = logistic_predict_proba(X, w, b)
    bce_loss = binary_cross_entropy(y, y_prob, w, l2_lambda)

    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)
    w_new = w - lr * dw
    b_new = b - lr * db

    return w_new, b_new, bce_loss

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...

    n_features = X.shape[1]
    w, b = initialize_logistic_params(n_features)


    losses = []

    for _ in range(n_epochs):

        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)

    return w, b, losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    return (proba >= threshold).astype(int)

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

