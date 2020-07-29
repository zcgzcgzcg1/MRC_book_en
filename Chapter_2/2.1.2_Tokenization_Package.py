# Install spaCy
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm')
text = ('Today is very special. I just got my Ph.D. degree.')
doc = nlp(text)
print([e.text for e in doc])

