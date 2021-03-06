# parse organic compound

import os
from scipy import sparse

restart = True
PARALLEL = 1  # assuming a quad-core machine
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

max_docs = 24



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
test_docs  = set()
splits = 5 / 6
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits * ld:
        train_docs.add(doc)
    else:
        test_docs.add(doc)
from pprint import pprint
pprint([x.name for x in train_docs])


from fonduer.snorkel.matchers import LambdaFunctionMatcher, Intersect, Union
from fonduer.snorkel.matchers import RegexMatchSpan

from regex_matcher import get_rgx_matcher

org_rgx = get_rgx_matcher()

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


from organic_spaces import OmniNgramsProd
prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

from fonduer.matchers import LambdaFunctionFigureMatcher

def white_black_list_matcher(fig):
    white_list = ['synthesis', 'plausible']
    black_list = ['spectra', 'x-ray', 'copyright', 'structur', 'application']

    fig_desc = fig.figure.description.lower()
    in_white = in_black = False
    if any(fig_desc.find(v) >= 0 for v in white_list): in_white = True
    if any(fig_desc.find(v) >= 0 for v in black_list): in_black = True
    if in_black and (not in_white):
        return False
    return True

def contain_organic_matcher(fig):
    # filter 2
    desc_wordlist = fig.figure.description.lower().split(' ')
    if any(re.search(org_rgx, w) for w in desc_wordlist): return True
    if not fig.figure.text == '':
        orc_wordlist = fig.figure.text.lower().split('\n')
        orc_wordlist = [w for w in orc_wordlist if not w == '']
        if any(re.search(org_rgx, w) for w in orc_wordlist): return True
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

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
candidate_extractor.apply(test_docs, split=1, parallelism=PARALLEL)

train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
test_cands = session.query(Org_Fig).filter(Org_Fig.split == 1).all()
print("Number of train candidates: {}\nNumber of test candidates: {}".format(len(train_cands), len(test_cands)))

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features

# Only need to do this once
print('Generating image features')
# session.execute("delete from context where stable_id like '%feature%'")
gen_image_features(docs_path=docs_path)

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
print('Generating other features')
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
F_test = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL) # generate sparse features
# print('Merging image features')
# F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0))  # concatenate dense with sparse matrix
# F_test = sparse.hstack(featurizer.load_matrix_and_image_features(split=1))  # concatenate dense with sparse matrix
#F_train = featurizer.load_matrix(split=0)
#F_test = featurizer.load_matrix(split=1)


from fonduer import BatchLabelAnnotator

# def LF_fig_name_match(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     product, img = args
#     if img.name == '':
#         return -1
#     else:
#         return 0
#
# def LF_match_page(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     return 1 if is_same_org_fig_page(organic, figure) else -1
#
#
# def LF_text_desc_match(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     product, img = args
#     if fuzz.partial_ratio(product.text, img.description) >= 70:
#         return 1
#     elif fuzz.partial_ratio(product.text, img.description) <= 40:
#         return -1
#     else:
#         return 0
#
#
# def LF_ocr_text_match(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     ocr_wordlist = figure.text.lower().split('\n')
#     ocr_wordlist = [w for w in ocr_wordlist if not w == '']
#     for w in ocr_wordlist:
#         if fuzz.partial_ratio(organic.text, w) >= 90:
#             return 1
#     return -1
#
#
# def LF_text_length_match(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     return -1 if len(organic.text) < 5 else 0
#
#
# def LF_match_whitelist(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     whitelist = ['synthesis', 'syntheses', 'made', 'catalyze', 'generate', 'product', 'produce',
#             'formation', 'developed', 'approach', 'yields', 'reaction', 'mechanism', 'proposed',
#             'fig', 'scheme', 'graph', 'diagram', 'table']
#     return 1 if both_contain_keywords(organic, figure, whitelist) else 0
#
# def LF_match_blacklist(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     blacklist = ['and', 'for', 'of', 'the', 'with', 'H2O', 'II']
#     return -1 if organic.text in blacklist else 0
#
#
# def LF_pos_near(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     return 1 if org_pos_near_fig(organic, figure) else 0
#
# def LF_organic_compound(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     return 1 if all([re.search(org_rgx, w) is not None for w in organic.text.split(' ')]) else 0
#
# def LF_synthesis_of(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     words = figure.description.split(' ')
#     words_lower = figure.description.lower().split(' ')
#     if organic.text in words and 'synthesis' in words_lower:
#         org_idx = words.index(organic.text)
#         syn_idx = words_lower.index('synthesis')
#         return 1 if syn_idx + 2 == org_idx else -1
#     return 0
#
# def LF_product_of(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     words = figure.description.split(' ')
#     words_lower = figure.description.lower().split(' ')
#     if organic.text in words and 'product' in words_lower:
#         org_idx = words.index(organic.text)
#         pro_idx = words_lower.index('product')
#         return 1 if pro_idx + 2 == org_idx else -1
#     return 0
#
# def LF_first_period(c):
#     args = c.get_contexts()
#     if len(args) != 2:
#         raise NotImplementedError("Only handles binary candidates currently")
#     organic, figure, = args
#     if '.' in figure.description:
#         first = figure.description.split('.')[0]
#         return 1 if organic.text in first else 0
#     return 0
#
from organic_lfs import *
org_fig_lfs = [
    LF_fig_name_match,
    LF_text_desc_match,
    LF_ocr_text_match,
    LF_text_length_match,
    LF_match_whitelist,
    LF_match_blacklist,
    LF_match_page,
    LF_pos_near,
    LF_organic_compound,
    LF_synthesis_of,
    LF_product_of,
    LF_first_period,
]


labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)

L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
#L_train = labeler.load_matrix(split=0)

print(L_train.shape)

L_train.get_candidate(session, 0)

