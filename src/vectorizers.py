import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import csr_matrix, lil_matrix, hstack
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def mean_w2v_vector(text: str, model: KeyedVectors) -> np.ndarray:
    token_vectors = [model.get_vector(token) for token in word_tokenize(text) if token in model]
    sum_vector = np.sum(token_vectors, axis=0)
    mean_vector = sum_vector / np.linalg.norm(sum_vector)
    return mean_vector

def concat_w2v(w2v_list: list[np.ndarray], tfidf_matrix: csr_matrix) -> lil_matrix:
    w2v_len = w2v_list[0].shape[0]
    new_tfidf_matrix = lil_matrix((tfidf_matrix.shape[0], tfidf_matrix.shape[1] + w2v_len), dtype=float)
    for i, vec in enumerate(tqdm(w2v_list)):
        concated_row = hstack([tfidf_matrix[i], vec])
        new_tfidf_matrix[i] = concated_row
    return new_tfidf_matrix

def get_tfidf_vector(tokens: list[str], text_id: int, matrix, tfidf) -> np.ndarray:
    tfidf_vector = np.ndarray(len(tokens))
    # a vector with all the terms
    text_raw_vector = matrix[text_id, :]
    # leaving only terms that are present in the text
    for i in range(len(tokens)):
        idx = tfidf.vocabulary_.get(tokens[i])
        val = matrix[text_id, idx] if idx else 0
        tfidf_vector[i] = val
    # vector with tfidf values of each word in the text
    return tfidf_vector