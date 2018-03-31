# parse organic compound

import os
import sys
os.system('bash ./set_env.sh')
PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/home/xiuyuan/private/839/fonduer_new/839_fonduer'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://postgres:112233@localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'

max_docs = float(2)
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer import Document, Phrase, Figure

docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (0.5, 0.75)
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

# Part two


from fonduer import RegexMatchSpan, DictionaryMatch, RegexMatchSplitEach,\
    LambdaFunctionMatcher, Intersect, Union

# prefix_rgx = '((meth|di|bi|tri|tetra|hex|hept|iso)?(benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
# suffix_rgx = '(.+(ane|ene|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine))'
# dashes_rgx = '(\w*(\-?\d+\'?,\d+\'?\-?|\-[a-z]+\-)\w*)'
# ions_rgx = '([A-Z]+[a-z]*\d*\+)'
# #abbr_rgx = '([A-Z|\-][A-Z|\-]+)'
prefix_rgx = '((meth|hex|hept|iso|benz|tetra|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
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



# Part 2.5
prefix_rgx = '((meth|hex|hept|iso|benz|tetra|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|ene|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine))'
dashes_rgx = '(\w*(\-?\d+\'?,\d+\'?\-?|\-[a-z]+\-)\w*)'
ions_rgx = '([A-Z]+[a-z]*\d*\+)'
#abbr_rgx = '([A-Z|\-][A-Z|\-]+)'
prod_matcher = RegexMatchSpan(rgx='|'.join([prefix_rgx, suffix_rgx, ions_rgx]),
                              longest_match_only=False, ignore_case=False)


from fonduer import CandidateExtractor

from fonduer.lf_helpers import *
import re

def candidate_filter(c):
    return True


from product_spaces import OmniNgramsProd
prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

from fonduer.matchers import LambdaFunctionFigureMatcher

def do_nothing_matcher(fig):
    return True

fig_matcher = LambdaFunctionFigureMatcher(func=do_nothing_matcher)
from fonduer import OmniFigures

figs = OmniFigures()

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter,
                        symmetric_relations=False)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)


attr_matcher = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
eeca_rgx = r'([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = r'(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = r'(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)

from organic_spaces import OmniNgramsPart, OmniNgramsTemp

part_ngrams = OmniNgramsPart(parts_by_doc=None, n_max=3)
attr_ngrams = OmniNgramsTemp(n_max=2)
from fonduer import CandidateExtractor

from fonduer.matchers import LambdaFunctionFigureMatcher

def do_nothing_matcher(fig):
    return True

def do_nothing_filter(c):
    (orgnic, attr) = c
    if same_table((part, attr)):
        return (is_horz_aligned((part, attr)) or is_vert_aligned((part, attr)))
    return True

from fonduer.lf_helpers import *
import re

def stg_temp_filter(c):
    (part, attr) = c
    if same_table((part, attr)):
        return (is_horz_aligned((part, attr)) or is_vert_aligned((part, attr)))
    return True

candidate_filter = stg_temp_filter

fig_matcher = LambdaFunctionFigureMatcher(func=do_nothing_matcher)



from fonduer import OmniFigures

figs = OmniFigures(type='png')

from fonduer import CandidateExtractor


candidate_extractor = CandidateExtractor(Org_Fig,
                        [org_ngrams, figs],
                        [organic_matcher, fig_matcher],
                        candidate_filter=do_nothing_filter)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)

train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
print("Number of candidates:", len(train_cands))
