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
log.info('small English NER spacy model has been loaded...')

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."
doc = nlp(text)

html_ner = spacy.displacy.render(doc, style="ent", jupyter=False)
log.info('Generating NER image in HTML format...')
output_path = Path(os.path.join("spacy_images/", "ner_sentence.html"))
log.info('Saving HTML NER image to file...')
output_path.open('w', encoding="utf-8").write(html_ner)
log.info('HTML NER image has been saved to file...')
