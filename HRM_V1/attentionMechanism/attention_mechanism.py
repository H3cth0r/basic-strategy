import numpy as np


# Representation of the words that encode semanting meaning. This are 3 dimensional space.
word_embeddings = {
        'she':      np.array([0.2, 0.9, 0.1, 0.5]),
        'likes':    np.array([0.8, 0.3, 0.7, 0.2]),
        'coffee':   np.array([0.4, 0.6, 0.3, 0.9]),
}


# Create Input Matrix
# Stack embeddings vertically to form the input matrix X
# This matrix will be used to calculate the Query(Q), Key(K) and Value(V)

X = np.vstack([
    word_embeddings['she'],
    word_embeddings['likes'],
    word_embeddings['coffee'],
])

# Define weight matrixs W_q, W_k and W_v
# This are like the translators that adapt our embeddings for attention.
# In prod, these are learned by the model
W_q = np.array([[0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.9]])
W_k = np.array([[0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.9]])
W_v = np.array([[0.8, 0.2, 0.1, 0.1],
                [0.2, 0.8, 0.2, 0.1],
                [0.1, 0.2, 0.8, 0.1],
                [0.1, 0.1, 0.1, 0.9]])

print(W_q)
print(W_k)
print(W_v)
print("="*30)

# Compute Q, K and V Matrices
# These transformation help the model understand context
Q = np.dot(X, W_q)
K = np.dot(X, W_k)
V = np.dot(X, W_v)
print(Q)
print(K)
print(V)
print("="*30)

# Calculate Scores
# Dot product of Q and the transpose of k yields the raw atytention scores
# measuring the relevance of each word to the others
scores = np.dot(Q, K.T)
print(scores)
print("="*30)

# To avoid excessively large values that could destabilize training
# We scale the scores by diiving by the square root of d_k
d_k = K.shape[1]
scaled_scores = scores / np.sqrt(d_k)
print(scaled_scores)
print("="*30)

# Apply softmax to obtain attention weights
# Converts them into probability distributions, hihglighting the importance of each word
exp_scores = np.exp(scaled_scores)
attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
print(attention_weights)
print("="*30)

# FInal output is computed by multiplying attention_weights by b.
# Output combines the information from the entire sequence in a way that considers the relevance of each word
output = np.dot(attention_weights, V)
print(output)
print("="*30)
