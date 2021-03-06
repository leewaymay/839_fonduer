'''

Candidate Extraction

'''



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


from fonduer import Document

# divide train/dev/test

docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
# dev_docs   = set()
test_docs  = set()
splits = 5/6
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits * ld:
        train_docs.add(doc)
    # elif i < splits[1] * ld:
    #     dev_docs.add(doc)
    else:
        test_docs.add(doc)
from pprint import pprint
pprint([x.name for x in train_docs])


from fonduer.snorkel.matchers import RegexMatchSpan, RegexMatchSplitEach,\
    DictionaryMatch, LambdaFunctionMatcher, Intersect, Union

from tutorials.organic_synthesis_figures import regex_matcher
org_rgx = regex_matcher.get_rgx_matcher()
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



from tutorials.organic_synthesis_figures.organic_spaces import OmniNgramsProd
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
        # print('Filtered by f1!')
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

    # print('Filtered by f2! Removed!')
    # print(fig.figure.name + " " + fig.figure.description)
    return False

fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
fig_matcher = Union(fig_matcher1, fig_matcher2)
# fig_matcher = LambdaFunctionFigureMatcher(func=white_black_list_matcher)

from fonduer.candidates import OmniDetailedFigures

figs = OmniDetailedFigures()

#### extract candidate ####

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

# candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
candidate_extractor.apply(test_docs, split=1, parallelism=PARALLEL)
