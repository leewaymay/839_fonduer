from fonduer.lf_helpers import *
from regex_matcher import *
import re
org_rgx = get_rgx_matcher()
import pandas as pd

def LF_fig_name_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    product, img = args
    if img.name == '':
        return -1
    else:
        return 0

def LF_match_page(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return -1 if not is_same_org_fig_page(organic, figure) else 0


def LF_text_desc_partial_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    product, img = args

    # TODO: increase ratio?
    if fuzz.partial_ratio(product.text, img.description) >= 85:
        return 1
    elif fuzz.partial_ratio(product.text, img.description) <= 40:
        return -1
    else:
        return 0

def LF_text_desc_full_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    product, img = args

    for word in img.description.split():
        if fuzz.ratio(product.text, word) >= 90:
            return 1
    return 0


def LF_ocr_text_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    ocr_wordlist = figure.text.lower().split('\n')
    ocr_wordlist = [w for w in ocr_wordlist if not w == '']
    for w in ocr_wordlist:
        if fuzz.partial_ratio(organic.text, w) >= 92:
            return 1
    return -1


def LF_text_length_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return -1 if len(organic.text) < 6 else 0


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


def LF_pos_near(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    return 1 if org_pos_near_fig(organic, figure) else 0

def LF_check_redundant_word_in_organic(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    keyword = ['synthesis', 'syntheses', 'product', 'reaction', 'the', 'of', 'for', ]
    for key in keyword:
        if key in organic.text:
            return -1
    return 0

def LF_keyword_of(c):
    keyword = ['synthesis', 'syntheses', 'product', 'reaction', ]
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    words = figure.description.split(' ')
    words_lower = figure.description.lower().split(' ')
    for key in keyword:
        if organic.text in words and key in words_lower:
            org_idx = words.index(organic.text)
            syn_idx = words_lower.index(key)
            return 1 if syn_idx + 2 == org_idx else -1
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

def LF_page_not_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    if abs(max(organic.sentence.page) - figure.page) > 1 or abs(min(organic.sentence.page) - figure.page) > 1:
        return -1
    else:
        return 0

def LF_dict_match(c):
    args = c.get_contexts()
    if len(args) != 2:
        raise NotImplementedError("Only handles binary candidates currently")
    organic, figure, = args
    org_dict = pd.read_csv('organic_dictionary.csv', header=None)
    for i in list(org_dict[0]):
        if i in figure.description and i in organic.text:
            return 1
    return 0


org_fig_lfs = [
    # LF_fig_name_match,
    # LF_text_desc_partial_match,
    LF_text_desc_full_match,
    LF_ocr_text_match,
    LF_text_length_match,
    LF_match_whitelist,
    LF_match_blacklist,
    LF_match_page,
    LF_pos_near,
    # LF_organic_compound,
    # LF_synthesis_of,
    # LF_product_of,
    LF_keyword_of,
    LF_first_period,
    LF_dict_match,
]

