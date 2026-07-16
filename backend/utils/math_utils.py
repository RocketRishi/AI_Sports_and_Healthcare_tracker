import numpy as np
def vector_length(v):
    return np.linalg.norm(v)

def normalize_vector(v):
    norm = vector_length(v)
    if norm == 0:
        return v
    return v / norm

def dot_product(a, b):
    return np.dot(a, b)