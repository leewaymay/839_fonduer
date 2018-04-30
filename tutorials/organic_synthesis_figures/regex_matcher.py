from fonduer.snorkel.matchers import LambdaFunctionMatcher, Intersect, Union
from fonduer.snorkel.matchers import RegexMatchSpan
import re
from fonduer.lf_helpers import *

def get_rgx_matcher():

    prefix_rgx = '(\(?((mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly).*)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydro(xy)?|amino|alk).+)'
    suffix_rgx = '(.+(ane|yl|adiene|atriene|kene|k?yne|anol|anediol|anetriol|anone|acid|amine|xide|dine?|(or?mone)|thiol|cine?|rine?|thine?|tone?)s?\)?)'

    dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
    comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
    inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'

    org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])

    return org_rgx

def get_organic_matcher():
    org_rgx = get_rgx_matcher()

    rgx_matcher = RegexMatchSpan(rgx=org_rgx, longest_match_only=True, ignore_case=False)
    blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
    prod_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)
    blacklist_rgx = ['methods?.?']
    prod_blacklist_rgx_lambda_matcher = LambdaFunctionMatcher(
        func=lambda x: all([re.match(r, x.text) is None for r in blacklist_rgx]), ignore_case=False)

    # prod_matcher = rgx_matcher
    return Intersect(rgx_matcher, prod_blacklist_lambda_matcher, prod_blacklist_rgx_lambda_matcher)

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

def get_fig_matcher():
    fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
    fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
    fig_matcher = Union(fig_matcher1, fig_matcher2)
    return fig_matcher

def candidate_filter(c):
    (organic, figure) = c
    if same_file(organic, figure):
        if mentionsFig(organic, figure) or mentionsOrg(figure, organic):
            return True