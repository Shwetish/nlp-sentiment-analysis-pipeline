
#-------------------
#02_pos_and_ner.py
#-----------------
#Demonstrates Part Of Speech tagging, Named entity recognition for locations (GPE), and date extraction using spaCy.
"""

from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Part 1: Geographical Entity Recognition (GPE) ---
text_food = """Kiran want to know the famous foods in each 
state of India. So, he opened Google and search for this question. Google showed that 
in Delhi it is Chaat, in Gujarat it is Dal Dhokli, in Tamilnadu it is 
Pongal, in Andhrapradesh it is Biryani, in Assam it is Papaya Khar, 
in Bihar it is Litti Chowkha and so on for all other states"""

doc_food = nlp(text_food)
geo_locations = [
    ent.text for ent in doc_food.ents if ent.label_ == "GPE"
]
print("--- GEOGRAPHICAL LOCATIONS ---")
print("Geographical location Names:", geo_locations)

# --- Part 2: Date Extraction ---
text_dates = """Sachin Tendulkar was born on 24 April 1973, Virat Kholi was 
born on 5 November 1988, Dhoni was born on 7 July 1981 
and finally Ricky ponting was born on 19 December 1974.


doc_dates = nlp(text_dates)
birth_dates = [
    ent.text for ent in doc_dates.ents if ent.label_ == "DATE"
]
print("\n--- EXTRACTED DATES ---")
print("All Birth Dates:", birth_dates)

# --- Part 3: POS Tagging Summary ---
doc_pos = nlp(text_food)
pos_counts = Counter([token.pos_ for token in doc_pos])

print("\n--- POS TAG COUNTS ---")
for pos, count in pos_counts.items():
    print(f"{pos}: {count}")