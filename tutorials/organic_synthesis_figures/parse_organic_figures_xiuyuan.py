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

docs_path = os.environ['FONDUERHOME'] + 'tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + 'tutorials/organic_synthesis_figures/data/pdf/'

# max_docs = float(10)
# doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
# corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
# #                           flatten=['sup', 'sub', 'small'],
# #                           ignore=['italic', 'bold'],
#                            blacklist=['style', 'script', 'meta', 'noscript'])

# corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer import Document

# get docs
docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (0.2, 0.9)
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

prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).+)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine|(or?mone)|thiol)\)?)'

dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'


org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])
rgx_matcher = RegexMatchSplitEach(rgx = org_rgx,
                              longest_match_only=False, ignore_case=False)

blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
prod_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)
blacklist_rgx = ['methods?.?']
prod_blacklist_rgx_lambda_matcher = LambdaFunctionMatcher(
    func=lambda x: all([re.match(r, x.text) is None for r in blacklist_rgx]), ignore_case=False)

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
import time

def white_black_list_matcher(fig):
    # print("enter filter 1")
    # enter_time = time.time()
    white_list = ['synthesis', 'plausible']
    black_list = ['spectra', 'x-ray', 'copyright', 'structur', 'application']

    fig_desc = fig.figure.description.lower()
    in_white = in_black = False
    if any(fig_desc.find(v) >= 0 for v in white_list): in_white = True
    if any(fig_desc.find(v) >= 0 for v in black_list): in_black = True
    if in_black and (not in_white):
        print('Filtered by f1!')
        return False
    # # print("{} has passed filter 1 in {} seconds!".format(fig.figure.name, time.time()-enter_time))
    # elif in_black:
    #     desc_wordlist = fig.figure.description.lower().split(' ')
    #     if any(re.search(org_rgx, w) for w in desc_wordlist): return True
    #     if not fig.figure.text == '':
    #         orc_wordlist = fig.figure.text.lower().split('\n')
    #         orc_wordlist = [w for w in orc_wordlist if not w == '']
    #         if any(re.search(org_rgx, w) for w in orc_wordlist): return True
    #
    #     print('Filtered by f2! Removed!')
    #     print(fig.figure.name + " " + fig.figure.description)
    #     return False
    return True

def contain_organic_matcher(fig):
    # print("{} has failed filter 1 in {} seconds!".format(fig.figure.name, time.time() - enter_time))
    # filter 2
    desc_wordlist = fig.figure.description.lower().split(' ')
    if any(re.search(org_rgx, w) for w in desc_wordlist): return True
    if not fig.figure.text == '':
        orc_wordlist = fig.figure.text.lower().split('\n')
        orc_wordlist = [w for w in orc_wordlist if not w == '']
        if any(re.search(org_rgx, w) for w in orc_wordlist): return True

    print('Filtered by f2! Removed!')
    print(fig.figure.name + " " + fig.figure.description)
    return False

fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
fig_matcher = Union(fig_matcher1, fig_matcher2)
# fig_matcher = LambdaFunctionFigureMatcher(func=white_black_list_matcher)

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


from fonduer.lf_helpers import *
from fuzzywuzzy import fuzz
import re


def LF_text_desc_match(c):
    product, img = c.get_context()
    if fuzz.partial_ratio(product.text, img.description) >= 70:
        return 1
    else:
        return 0

def LF_ocr_text_match(c):
    product, img = c.get_context()
    ocr_wordlist = img.text.lower().split('\n')
    ocr_wordlist = [w for w in ocr_wordlist if not w == '']
    for w in ocr_wordlist:
        if fuzz.partial_ratio(product.text, w) >= 90:
            return 1
    return 0

def LF_text_lenth_match(c):
    product, img = c.get_context()
    return -1 if len(product.text) < 5 else 0

def LF_text_lenth_match(c):
    product, img = c.get_context()
    return -1 if len(product.text) < 5 else 0


