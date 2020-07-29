# Named entity recognition and part-of-speech tagging
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"Apple may buy a U.K. startup for $1 billion")
print('-----Part of Speech-----')
for token in doc:
    print(token.text, token.pos_)

print('-----Named Entity Recognition-----')
for ent in doc.ents:
    print(ent.text, ent.label_)
