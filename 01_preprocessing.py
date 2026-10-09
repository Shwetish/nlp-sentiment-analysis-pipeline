#----------------------------------------------------
#01_preprocessing.py
#----------------------------------------------------
#Demonstrates text tokenization, Porter Stemming, and WordNet Lemmatization using NLTK.

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize

# --- Part 1: Tokenize with RegexpTokenizer ---
transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"
tokenizer = RegexpTokenizer(r"\w+|[\$\€]")
all_tokens = tokenizer.tokenize(transactions)

target_outputs = ["two", "$", "500", "€"]
print("--- extracted targets ---")
for token in all_tokens:
    if token in target_outputs:
        print(token)
        
######################################################
# --- Part 2: Stemming vs. Lemmatization ---
text = """Latha is very multi talented girl.She is good at many skills like dancing, running, singing, playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this, she is a wonderful at cooking too."""

tokens = word_tokenize(text.lower())
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in tokens]
lemmatizer_words = [lemmatizer.lemmatize(word) for word in tokens]

print("\n--- STEMMING VS LEMMATIZATION ---")
print(
    f"{'Original Word':<15} | {'Stemmed (Porter)':<15} | {'Lemmatized (WordNet)':<15}"
)
print("-" * 53)

for orig, stem, lem in zip(tokens, stemmed_words, lemmatizer_words):
    if orig.isalnum():
        print(f"{orig:<15} | {stem:<15} | {lem:<15}")