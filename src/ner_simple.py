from utils.logger import get_log_object
import spacy
from spacy import displacy
from pathlib import Path
import os


# instantiate log
log = get_log_object()

# Read in data
log.info('Simple example for NER using spacy...')

nlp = spacy.load("en_core_web_sm")

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
doc = nlp(text)

html_ner = spacy.displacy.render(doc, style="ent", jupyter=False)
output_path = Path(os.path.join("../spacy_images/", "ner_sentence.html"))
output_path.open('w', encoding="utf-8").write(html_ner)