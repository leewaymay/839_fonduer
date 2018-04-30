import os
import sys
from scipy import sparse

PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/Users/liwei/BoxSync/s2016/Dropbox/839_fonduer'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])


docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features

#Only need to do this once
gen_image_features(docs_path=docs_path)

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
print('merging results')
F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0))


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


prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).*)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydro(xy)?|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine|(or?mone)|thiol|cin)\)?)'

dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'

org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])

def LF_organic_compound(c):
    args = c.get_contexts()
    organic = args[0]
    result = re.search(org_rgx, organic.text)
    return 1 if result else 0


def LF_fig_name_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    product, img = args
    if img.name == '':
        return -1
    else:
        return 0


def LF_text_desc_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    product, img = args
    if fuzz.partial_ratio(product.text, img.description) >= 70:
        return 1
    elif fuzz.partial_ratio(product.text, img.description) <= 40:
        return -1
    else:
        return 0


def LF_ocr_text_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    ocr_wordlist = figure.text.lower().split('\n')
    ocr_wordlist = [w for w in ocr_wordlist if not w == '']
    for w in ocr_wordlist:
        if fuzz.partial_ratio(organic.text, w) >= 90:
            return 1
    return -1


def LF_text_length_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return -1 if len(organic.text) < 5 else 0


def LF_match_whitelist(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    whitelist = ['synthesis', 'syntheses', 'made', 'catalyze', 'generate', 'product', 'produce',
            'formation', 'developed', 'approach', 'yields', 'reaction', 'mechanism', 'proposed',
            'fig', 'scheme', 'graph', 'diagram', 'table']
    return 1 if both_contain_keywords(organic, figure, whitelist) else 0

def LF_match_blacklist(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    blacklist = ['and', 'for', 'of', 'the', 'with', 'H2O', 'II']
    return -1 if organic.text in blacklist else 0

def LF_synthesis_of(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    words = figure.description.split(' ')
    words_lower = figure.description.lower().split(' ')
    if organic.text in words and 'synthesis' in words_lower:
        org_idx = words.index(organic.text)
        syn_idx = words_lower.index('synthesis')
        return 1 if syn_idx + 2 == org_idx else -1
    return 0

def LF_product_of(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    words = figure.description.split(' ')
    words_lower = figure.description.lower().split(' ')
    if organic.text in words and 'product' in words_lower:
        org_idx = words.index(organic.text)
        pro_idx = words_lower.index('product')
        return 1 if pro_idx + 2 == org_idx else -1
    return 0

def LF_first_period(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    if '.' in figure.description:
        first = figure.description.split('.')[0]
        return 1 if organic.text in first else 0
    return 0

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

from fonduer import BatchLabelAnnotator

labeler = BatchLabelAnnotator(Org_Fig, lfs = org_fig_lfs)
L_train = labeler.load_matrix(split=0)

# L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
print(L_train.shape)

L_train.get_candidate(session, 0)

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)

train_marginals = gen_model.marginals(L_train)
print(gen_model.weights.lf_accuracy)

from fonduer import SparseLogisticRegression


disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

#Current we only predict on the training set
test_score = disc_model.predictions(F_train)
