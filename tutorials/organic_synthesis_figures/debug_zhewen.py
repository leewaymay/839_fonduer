# parse organic compound

import os
from scipy import sparse

reparse = False  # VERY EXPENSIVE
reextract = False
refeaturize = False  # VERY EXPENSIVE
with_image_feats = True
relabel = False

PARALLEL = 1  # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/Users/Zitman/Documents/Graduate/Courses/CS839/Project/839_fonduer/tutorials'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/data/pdf/'

max_docs = 24
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
#                           flatten=['sup', 'sub', 'small'],
#                           ignore=['italic', 'bold'],
                           blacklist=['style', 'script', 'meta', 'noscript'])

if reparse:
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


from fonduer.snorkel.matchers import LambdaFunctionMatcher, DictionaryMatch, Intersect, Union
from fonduer.snorkel.matchers import RegexMatchSpan

from regex_matcher import get_rgx_matcher

org_rgx = get_rgx_matcher()

rgx_matcher = RegexMatchSpan(rgx=org_rgx, longest_match_only=True, ignore_case=False)
blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
org_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)
blacklist_rgx = ['methods?.?']
org_blacklist_rgx_lambda_matcher = LambdaFunctionMatcher(
    func=lambda x: all([re.match(r, x.text) is None for r in blacklist_rgx]), ignore_case=False)

import csv

def get_organic_set(path):
    all_orgs = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            (org, _) = line
            all_orgs.add(org)
    return all_orgs

dict_path = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/organic_dictionary.csv'
org_dict_matcher = DictionaryMatch(d=get_organic_set(dict_path))

org_matcher = Union(org_dict_matcher,
                    Intersect(rgx_matcher, org_blacklist_lambda_matcher, org_blacklist_rgx_lambda_matcher))

from fonduer import CandidateExtractor
from fonduer.lf_helpers import *
import re

def candidate_filter(c):
    (organic, figure) = c
    to_remove = ['synthesis', 'syntheses', 'product', 'reaction', 'the', 'of', 'for', ]
    for key in to_remove:
        if key in organic.text.split():
            return False
    if same_file(organic, figure):
        if mentionsFig(organic, figure) or mentionsOrg(figure, organic):
            return True


from organic_spaces import OmniNgramsProd
org_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

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
                        [org_ngrams, figs],
                        [org_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

if reextract:
    candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
    candidate_extractor.apply(test_docs, split=1, parallelism=PARALLEL)

train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
test_cands = session.query(Org_Fig).filter(Org_Fig.split == 1).all()
print("Number of train candidates: {}\nNumber of test candidates: {}".format(len(train_cands), len(test_cands)))

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features
featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)

if refeaturize:
    # Only need to do this once
    print('Generating image features')
    session.execute("delete from context where stable_id like '%feature%'")
    gen_image_features(docs_path=docs_path)
    print('Generating other features')
    F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
    F_test = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL) # generate sparse features

if with_image_feats:
    print('Merging image features')
    F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0)).toarray()  # concatenate dense with sparse matrix
    F_test = sparse.hstack(featurizer.load_matrix_and_image_features(split=1)).toarray()  # concatenate dense with sparse matrixs
else:
    F_train = featurizer.load_matrix(split=0).toarray()
    F_test = featurizer.load_matrix(split=1).toarray()

from fonduer import BatchLabelAnnotator

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
    LF_check_redundant_word_in_organic,
    LF_keyword_of,
    LF_first_period,
]


labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)

if relabel:
    L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
else:
    L_train = labeler.load_matrix(split=0)

print(L_train.shape)

L_train.get_candidate(session, 0)

L_test = labeler.apply_existing(split=1)

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
print(gen_model.weights.lf_accuracy)

train_marginals = gen_model.marginals(L_train)

from fonduer import LogisticRegression

disc_model = LogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

F_train_sparse = featurizer.load_matrix(split=0)
F_test_sparse = featurizer.load_matrix(split=1)
F_test_sparse.get_candidate(session, 0)

test_candidates = [F_test_sparse.get_candidate(session, i) for i in range(F_test_sparse.shape[0])]
test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]
train_score = disc_model.predictions(F_train)

# load gold label
from tutorials.organic_synthesis_figures.organic_utils import load_organic_labels
from fonduer import load_gold_labels

gold_file = os.environ['FONDUERHOME'] + '/organic_synthesis_figures/organic_gold.csv'
load_organic_labels(session, Org_Fig, gold_file, ATTRIBUTE ,annotator_name='gold')

L_gold_train = load_gold_labels(session, annotator_name="gold", split=0)
print(L_train.lf_stats(L_gold_train))

L_gold_test = load_gold_labels(session, annotator_name="gold", split=1)

prec, rec, f1 = gen_model.score(L_test, L_gold_test)

from organic_utils import entity_level_f1

test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]

(TP, FP, FN) = entity_level_f1(true_pred, gold_file, ATTRIBUTE, test_docs)


from matplotlib import pyplot as plt

def plot_tp_entity(e):
    fig = plt.Figure(figsize=(20,20))
    im = plt.imread(docs_path+e[0])
    plt.imshow(im, cmap='gray')
    plt.title("Gold: {}, Extracted: {}".format(e[2], e[1]))
    print(e[0])

plot_tp_entity(TP[0])