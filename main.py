import re
from collections import Counter

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

