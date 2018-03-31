# parse organic compound

import os
import sys
# os.system('bash ./set_env.sh')
PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/home/xiuyuan/private/839/fonduer_new/839_fonduer/'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://postgres:112233@localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['product','figure'])

from fonduer import HTMLPreprocessor, OmniParser


docs_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/sandbox/html/'
pdf_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/sandbox/pdf/'

max_docs = 3
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
#                           flatten=['sup', 'sub', 'small'],
#                           ignore=['italic', 'bold'],
                           blacklist=['style', 'script', 'meta', 'noscript'])

#corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer import Document

# get docs
docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (0.8, 0.9)
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


from fonduer.snorkel.matchers import RegexMatchSpan, RegexMatchSplitEach,\
    DictionaryMatch, LambdaFunctionMatcher, Intersect, Union


prefix_rgx = '(.+(meth|cycl|tri|tetra|hex|hept|iso|carb|benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine).+)'

dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'


rgx_matcher = RegexMatchSplitEach(rgx='|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx]),
                              longest_match_only=True, ignore_case=False)

blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
prod_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)

#prod_matcher = rgx_matcher
prod_matcher = Intersect(rgx_matcher, prod_blacklist_lambda_matcher)

from fonduer import CandidateExtractor
from fonduer.lf_helpers import *
import re

def candidate_filter(c):
    (organic, figure) = c
    if same_file(organic, figure):
        if mentionsFig(organic, figure) or mentionsOrg(figure, organic):
            return True


from tutorials.organic_synthesis_figures.product_spaces import OmniNgramsProd
prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

from fonduer.matchers import LambdaFunctionFigureMatcher

def white_black_list_matcher(fig):
    white_list = ['synthesis', 'plausible']
    black_list = ['spectra', 'x-ray', 'copyright']

    fig_desc = fig.figure.description.replace(" ", "").lower()
    w = b = False
    if any(fig_desc.find(v) >= 0 for v in white_list): w = True
    if any(fig_desc.find(v) >= 0 for v in black_list): b = True
    if b and (not w):
        print('removed due to black list and not saved by white list!')
        print(fig.figure.description)
        return False
    return True

def contain_organic_matcher(fig):


    # fig_desc_wlist = fig.figure.description.split(" ")
    fig_desc = fig.figure.description.replace(" ", "").lower()
    # fig_desc_wlist = [word.lower() for word in fig_desc_wlist]
    # ocr_wlist = fig.figure.text.split("\n")

    orc_str = fig.figure.text.lower()
    if re.search(org_rgx, fig_desc) or re.search(org_rgx, orc_str):
        return True
    # for word in ocr_wlist:
    #     if not word or len(word.strip()) == 0:
    #         continue
    #     wl = word.strip().split(" ")
    #     wl = [w.lower() for w in wl]
    #     for v in wl:
    #         if prod_matcher.f(v):
    #             return True
    print('removed due to not contain organic!')
    print(fig.figure.description)
    return False


fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
fig_matcher = Union(fig_matcher1, fig_matcher2)

from fonduer.candidates import OmniDetailedFigures

figs = OmniDetailedFigures()

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)

train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
print("Number of candidates:", len(train_cands))

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL)