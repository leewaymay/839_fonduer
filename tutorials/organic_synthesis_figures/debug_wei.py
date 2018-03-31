import os
import sys

PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/Users/liwei/BoxSync/s2016/Dropbox/839_fonduer'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'

from fonduer import Document, Phrase, Figure

docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (1, 1)
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits[0] * ld:
        train_docs.add(doc)
    elif i < splits[1] * ld:
        dev_docs.add(doc)
    else:
        test_docs.add(doc)
from pprint import pprint
pprint([x.name for x in train_docs])

from fonduer import RegexMatchSpan, DictionaryMatch, RegexMatchSplitEach,\
    LambdaFunctionMatcher, Intersect, Union

prefix_rgx = '((meth|di|bi|tri|tetra|hex|hept|iso)?(benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|ene|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine))'
dashes_rgx = '(\w*(\-?\d+\'?,\d+\'?\-?|\-[a-z]+\-)\w*)'
ions_rgx = '([A-Z]+[a-z]*\d*\+)'
#abbr_rgx = '([A-Z|\-][A-Z|\-]+)'


prod_matcher = RegexMatchSplitEach(rgx='|'.join([prefix_rgx, suffix_rgx, ions_rgx]),
                              longest_match_only=False, ignore_case=False)


from fonduer import CandidateExtractor

from fonduer.lf_helpers import *
import re

def candidate_filter(c):
    (organic, figure) = c
    if same_file(organic, figure):
        if mentionsFig(organic, figure) or mentionsOrg(figure, organic):
            return True


from product_spaces import OmniNgramsProd
prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

from fonduer.matchers import LambdaFunctionFigureMatcher

def do_nothing_matcher(fig):
    return True

fig_matcher = LambdaFunctionFigureMatcher(func=do_nothing_matcher)
from fonduer.candidates import OmniDetailedFigures

figs = OmniDetailedFigures()

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)

import timeit
timeit.timeit('print("hello!")')