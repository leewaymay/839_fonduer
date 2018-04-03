# parse organic compound

import os
import sys

PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/Users/Zitman/Documents/Graduate/Courses/CS839/Project/839_fonduer/tutorials'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']

restart = False

from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/pdf/'

max_docs = 10
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
#                           flatten=['sup', 'sub', 'small'],
#                           ignore=['italic', 'bold'],
                           blacklist=['style', 'script', 'meta', 'noscript'])

if restart:
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from fonduer import Document

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

prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).+)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine|(or?mone)|thiol)\)?)'

dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'

org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])

rgx_matcher = RegexMatchSpan(rgx=org_rgx, longest_match_only=True, ignore_case=False)

blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
prod_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)
blacklist_rgx = ['methods?.?']
prod_blacklist_rgx_lambda_matcher = LambdaFunctionMatcher(
    func=lambda x: all([re.match(r, x.text) is None for r in blacklist_rgx]), ignore_case=False)

#prod_matcher = rgx_matcher
prod_matcher = Intersect(rgx_matcher, prod_blacklist_lambda_matcher, prod_blacklist_rgx_lambda_matcher)

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
        return False
    # print("{} has passed filter 1 in {} seconds!".format(fig.figure.name, time.time()-enter_time))
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

    #print(fig.figure.name + " " + fig.figure.description)
    return False

fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
fig_matcher = Union(fig_matcher1, fig_matcher2)
# fig_matcher = LambdaFunctionFigureMatcher(func=figure_filter)


from fonduer.candidates import OmniDetailedFigures

figs = OmniDetailedFigures()

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

if restart:
    candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)

train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
print("Number of candidates:", len(train_cands))

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL)

from fonduer import BatchLabelAnnotator

whitelist = ['synthesis', 'syntheses', 'made', 'catalyze', 'generate', 'product', 'produce',
            'formation', 'developed', 'approach', 'yields', 'reaction',
            'fig', 'scheme', 'graph', 'diagram', 'table']

blacklist = ['and', 'for', 'of', 'the']
def LF_same_file(c):
    return int(c.organic.sentence.document == c.figure.document)

lf_org_img = [LF_same_file] + [lambda c: 1 if kw in c.organic.sentence.text else 0 for kw in whitelist]
lf_org_img += [lambda c: 0 if kw in c.organic.words else 1 for kw in blacklist]

labeler = BatchLabelAnnotator(Org_Fig, lfs = lf_org_img)
L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)

print(L_train.shape)

L_train.get_candidate(session, 0)

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
train_marginals = gen_model.marginals(L_train)

from fonduer import SparseLogisticRegression

disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)