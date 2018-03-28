# parse organic compound

import os
import sys

PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/Users/Zitman/Documents/Graduate/Courses/CS839/Project/839_fonduer'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/pdf/'

max_docs = float(1)
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
#                           flatten=['sup', 'sub', 'small'],
#                           ignore=['italic', 'bold'],
#                           blacklist=['style', 'script', 'meta', 'noscript'])
)
corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer import Phrase

for p in session.query(Phrase).all():
    print(p.text)
