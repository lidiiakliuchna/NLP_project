import re
from collections import defaultdict, Counter
from string import punctuation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_language_sentences(file_name, language):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Define a regular expression pattern 
        pattern = re.compile(f'^\d+\t{language}\t([^\t]+)')

        # Extract sentences
        language_sentences = [re.match(pattern, line).group(1) for line in lines if re.match(pattern, line)]

        return language_sentences

    except Exception as e:
        print(f"Error: {e}")
        return None

file_name = 'sentences_detailed.csv'
language_english = 'eng'
language_russian = 'rus'
language_german = 'deu'

# Extract sentences for selected languages
english_sentences = extract_language_sentences(file_name, language_english)
russian_sentences = extract_language_sentences(file_name, language_russian)
german_sentences = extract_language_sentences(file_name, language_german)

def clean_text(text):
    # Remove special characters
    cleaned_text = re.sub(r'\xa0', ' ', text)
    # Remove punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Remove spaces
    cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()
    return cleaned_text


# Clean each sentence
cleaned_russian_sentences = [clean_text(sentence) for sentence in russian_sentences]
cleaned_english_sentences = [clean_text(sentence) for sentence in english_sentences]
cleaned_german_sentences = [clean_text(sentence) for sentence in german_sentences]

# Creating n-grams from a string
def n_grams(a, n):
    return [a[i:i+n] for i in range(len(a)-n+1)]

# Calculate character n-gram frequencies
lang2char_ngrams_freqs = defaultdict(Counter)
table = str.maketrans({ch: None for ch in punctuation})

for lang, lang_sentences in zip(['Russian', 'English', 'German'], [cleaned_russian_sentences, cleaned_english_sentences, cleaned_german_sentences]):
    for text in lang_sentences:
        char_ngrams = n_grams(text.lower().translate(table), 3)
        lang2char_ngrams_freqs[lang].update(filter(bool, char_ngrams))

# Create sets of top 100 character n-grams
lang2char_ngrams = {}
for lang in lang2char_ngrams_freqs:
    topn = [word for word, freq in lang2char_ngrams_freqs[lang].most_common(100)]
    lang2char_ngrams[lang] = set(topn)

# Predict language based on character n-grams
def predict_language(text, lang2char):
    table = str.maketrans({ch: None for ch in punctuation})
    text_ngrams = set(n_grams(text.lower().translate(table), 3))

    lang2sim = {}

    for lang, char_ngrams_set in lang2char.items():
        intersect = len(text_ngrams & char_ngrams_set)
        lang2sim[lang] = intersect

    return max(lang2sim, key=lambda x: lang2sim[x])

# Testing
predicted_lang = predict_language('Das ist fantastisch ', lang2char_ngrams)
print(predicted_lang)

# Predict languages for all texts
true_labels = []
predicted_labels = []

for lang, lang_sentences in zip(['Russian', 'English', 'German'], [cleaned_russian_sentences, cleaned_english_sentences, cleaned_german_sentences]):
    for text in lang_sentences:
        true_labels.append(lang)
        predicted_labels.append(predict_language(text, lang2char_ngrams))

# Print classification report
print(classification_report(true_labels, predicted_labels))

# Print confusion matrix
labels = list(set(true_labels))
print(confusion_matrix(true_labels, predicted_labels, labels=labels))

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(data=confusion_matrix(true_labels, predicted_labels, labels=labels),
            annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
plt.title("Confusion matrix")
plt.show()

# Create a common vocabulary of n-grams
vocab = Counter()

# Update the vocabulary with n-grams 
for lang_sentences in [cleaned_russian_sentences, cleaned_english_sentences, cleaned_german_sentences]:
    for text in lang_sentences:
        vocab.update(n_grams(text.lower(), 2))

# Select top 5000 n-grams
vocab = [word for word, _ in vocab.most_common(5000)]

# Create n-gram to index and index to n-gram dictionaries
id2ngram = {i: ngram for i, ngram in enumerate(vocab)}
ngram2id = {ngram: i for i, ngram in enumerate(vocab)}

# Represent each language as a vector of n-gram frequencies
lang2vec = defaultdict(lambda: np.zeros((len(vocab))))


for lang, lang_sentences in zip(['Russian', 'English', 'German'], [cleaned_russian_sentences, cleaned_english_sentences, cleaned_german_sentences]):
    for text in lang_sentences:
        ngrams = n_grams(text.lower(), 2)
        for ngram in ngrams:
            if ngram in ngram2id:
                lang2vec[lang][ngram2id[ngram]] += 1


# Predict language based on cosine similarity
def predicted_language(text, lang2vec):
    table = str.maketrans({ch: None for ch in punctuation})
    text_ngrams = n_grams(text.lower().translate(table), 2)
    text_vec = np.zeros((len(vocab)))

    
    for ngram in text_ngrams:
        if ngram in ngram2id:
            text_vec[ngram2id[ngram]] += 1

    lang2sim = {}

    # Compute cosine similarity between input text vector and language vectors
    for lang, lang_vec in lang2vec.items():
        similarity = cosine_similarity([text_vec], [lang_vec])[0][0]
        lang2sim[lang] = similarity

    # Predict the language with the highest similarity
    return max(lang2sim, key=lambda x: lang2sim[x])


# Test the predict_language function
predicted_lang_1 = predicted_language('Das ist fantastisch ', lang2vec)
print(predicted_lang_1)

# Predict languages for all texts
true_labels = []
predicted_labels = []

for lang, lang_sentences in zip(['Russian', 'English', 'German'], [cleaned_russian_sentences, cleaned_english_sentences, cleaned_german_sentences]):
    for text in lang_sentences:
        true_labels.append(lang)
        predicted_labels.append(predicted_language(text, lang2vec))

# Print classification report
print(classification_report(true_labels, predicted_labels))

# Print confusion matrix
labels = list(set(true_labels))
print(confusion_matrix(true_labels, predicted_labels, labels=labels))

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(data=confusion_matrix(true_labels, predicted_labels, labels=labels),
            annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
plt.title("Confusion matrix")
plt.show()
