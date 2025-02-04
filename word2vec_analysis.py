import gensim
import sklearn
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns # type: ignore

# ✅ Step 1: Train a Word2Vec Model on a Larger Corpus
# Expanding the corpus with more sentences
corpus = [
    "The king rules the kingdom",
    "The queen is the wife of the king",
    "A cat is an animal",
    "Dogs and cats are common pets",
    "The man and woman are human beings",
    "A child plays with toys",
    "The economy affects the government",
    "Politics and democracy are related",
    "The lion is the king of the jungle",
    "The dog barks at strangers",
    "Children love playing outdoors",
    "Democracy ensures freedom of speech",
    "The politician made a powerful speech",
    "Economics plays a crucial role in politics"
]

# Preprocess and tokenize
sentences = [sentence.lower().split() for sentence in corpus]

# Train the Word2Vec model with Skip-gram and higher vector size
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)
model.save("models/word2vec.model")
print("✅ Word2Vec Model trained and saved successfully!")

# ✅ Step 2: Word Similarity Evaluation
def evaluate_word_similarity(word1, word2):
    try:
        similarity = model.wv.similarity(word1, word2)
        print(f"🔹 Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    except KeyError as e:
        print(f"⚠️ Error: {e} (One or both words not in vocabulary)")

evaluate_word_similarity("king", "queen")
evaluate_word_similarity("cat", "dog")
evaluate_word_similarity("politics", "government")



# ✅ Step 3: Intrinsic Evaluation - Word Analogy Test
def analogy(word_a, word_b, word_c):
    try:
        result = model.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=1)
        print(f"🔹 {word_a} → {word_b}, {word_c} → {result[0][0]}")
    except KeyError as e:
        print(f"⚠️ Error: {e} (Words not in vocabulary)")

analogy("king", "queen", "man")  # Expected: woman
analogy("dog", "puppy", "cat")   # Expected: kitten

# ✅ Step 4: Extrinsic Evaluation - Sentiment Analysis (Dummy Example)
# Using embeddings to train a classifier (binary classification)
X = []
y = []

# Example training sentences with labels (1: Positive, 0: Negative)
train_data = [
    ("The food was amazing and delicious", 1),
    ("I hated the movie, it was terrible", 0),
    ("The book was fantastic, I loved it", 1),
    ("The service was bad and rude", 0),
    ("The weather is wonderful today", 1),
    ("I dislike the taste of this coffee", 0)
]

# Updated for checking word presence in model vocabulary
for sentence, label in train_data:
    words = sentence.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:  # Only process sentences that have words in the vocabulary
        X.append(np.mean(word_vectors, axis=0))  # Take average of word embeddings
        y.append(label)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Sentiment Classification Accuracy: {accuracy:.4f}")

# ✅ Step 5: Visualization (PCA & t-SNE)
def visualize_embeddings(method="PCA"):
    words = list(model.wv.index_to_key)[:20]  # Select first 20 words
    vectors = np.array([model.wv[word] for word in words])

    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=5)  # Set perplexity to a lower value

    reduced_vectors = reducer.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title(f"Word Embeddings Visualization using {method}")
    plt.show()

# Visualizing embeddings
print("Visualizing embeddings...")
visualize_embeddings("t-SNE")

