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





from fonduer.lf_helpers import *
import re


def LF_match_keywords(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    keywords = ['synthesis', 'reaction', 'produce', 'yield', 'formation', 'approach']
    return 1 if both_contain_keywords(organic, figure, keywords) else 0

def LF_match_page(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return 1 if is_same_org_fig_page(organic, figure) else 0

def LF_pos_near(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return 1 if org_pos_near_fig(organic, figure) else 0

prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).+)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine|(or?mone)|thiol)\)?)'

dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'

org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])


def LF_organic_compound(c):
    args = c.get_contexts()
    organic = args[0]
    result = re.search(org_rgx, organic.text)
    return 1 if result else 0




# def LF_temperature_row(c):
#     return 1 if 'temperature' in get_row_ngrams(c.attr) else 0
#
#
# def LF_operating_row(c):
#     return 1 if 'operating' in get_row_ngrams(c.attr) else 0
#
# def LF_tstg_row(c):
#     return 1 if overlap(
#         ['tstg','stg','ts'],
#         list(get_row_ngrams(c.attr))) else 0
#
#
# def LF_to_left(c):
#     return 1 if 'to' in get_left_ngrams(c.attr, window=2) else 0
#
# def LF_negative_number_left(c):
#     return 1 if any([re.match(r'-\s*\d+', ngram) for ngram in get_left_ngrams(c.attr, window=4)]) else 0


org_fig_lfs = [
    LF_match_keywords,
    LF_match_page,
    LF_pos_near,
    LF_organic_compound
]


from fonduer import BatchLabelAnnotator

labeler = BatchLabelAnnotator(Org_Fig, lfs = org_fig_lfs)
L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
print(L_train.shape)

L_train.get_candidate(session, 0)

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)

train_marginals = gen_model.marginals(L_train)
print(gen_model.weights.lf_accuracy)

from fonduer import SparseLogisticRegression
from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
F_train = featurizer.load_matrix(split=0)

disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

#Current we only predict on the training set
test_score = disc_model.predictions(F_train)
