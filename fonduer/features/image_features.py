from fonduer.models import ImplicitSpan
from fonduer.snorkel.models import TemporarySpan
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from fonduer import lf_helpers
import re

prefix_rgx = '(\(?(mono|bi|di|tri|tetra|hex|hept|oct|iso|a?cycl|poly)?(meth|carb|benz|fluoro|chloro|bromo|iodo|hydroxy|amino|alk).+)'
suffix_rgx = '(.+(ane|yl|adiene|atriene|yne|anol|anediol|anetriol|anone|acid|amine|xide|dine|(or?mone)|thiol)\)?)'
dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?\-)\w*)'
comma_dash_rgx = '((\w+\-|\(?)([a-z|\d]\'?,[a-z|\d]\'?\-)\w*)'
inorganic_rgx = '(([A-Z][a-z]?\d*\+?){2,})'
org_rgx = '|'.join([prefix_rgx, suffix_rgx, dash_rgx, comma_dash_rgx, inorganic_rgx])

prefix_list = ['cycl', 'tri', 'tetra', 'hex', 'hept', 'iso', 'carb', 'benz', 'fluoro', 'chloro', 'bromo', 'iodo',
               'hydroxy', 'amino', 'alk']
suffix_list = ['ane', 'yl', 'adiene', 'atriene', 'yne', 'anol', 'anediol', 'anetriol', 'anone', 'acid', 'amine', 'xide',
               'dine']
keyword_list = ['synthesis', 'syntheses', 'made', 'generate', 'reaction', 'product', 'yield', 'formation', 'produce',
                'catalyze', 'developed', 'approach']
black_word_list = ['x-ray', 'copyright']

FEAT_PRE = "IMAGE_"
DEF_VALUE = 1

unary_feats = {}


def get_image_feats(candidates):


    candidates = candidates if isinstance(candidates, list) else [candidates]

    desc_sum = text_sum = 0
    for candidate in candidates:
        img = candidate.get_contexts()[1]
        desc_sum += len(img.description)
        text_sum += len(img.text.replace('\n', '').replace(' ', ''))
    desc_avg = float(desc_sum / len(candidates))
    text_avg = float(text_sum / len(candidates))

    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." %
                             type(candidate))
        # Unary candidates
        # if len(args) == 1:
        #     span = args[0]
        #
        #     if span.stable_id not in unary_feats:
        #         unary_feats[span.stable_id] = set()
        #         for f in _generate_img_feats(span, desc_avg = desc_avg, text_avg = text_avg):
        #             unary_feats[span.stable_id].add(f)
        #
        #     for f in unary_feats[span.stable_id]:
        #         yield candidate.id, FEAT_PRE + f, DEF_VALUE

        # Binary candidates
        if len(args) == 2:
            span, pre = args[1], 'e2_'
            if span.stable_id not in unary_feats:
                unary_feats[span.stable_id] = set()

                for f in _generate_img_feats(span, desc_avg = desc_avg, text_avg = text_avg):
                    unary_feats[span.stable_id].add(f)

            for f in unary_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + pre + f, DEF_VALUE
        else:
            raise NotImplementedError(
                "Only handles binary candidates for product/image currently")


def _generate_img_feats(span, **kwargs):
    # core features
    yield "SPAN_TYPE_[%s]" % (
        'IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT')

    yield "IMG_TYPE_[%s]" % (
        'SCHEME' if not span.name.lower().find('scheme') == -1 else 'FIGURE')

    # length of image description and ocr text
    yield "IMG_DESC_[%s]_THAN_AVG" % ('GREATER' if len(span.description) >= kwargs.get('desc_avg') else 'SMALLER')

    ocr_flag = False

    if len(span.text) == 0:
        yield "IMG_TEXT_EMPTY"
    else:
        ocr_purestring = span.text.replace('\n', "").replace(" ", "").replace("/", '').replace("\\", '')
        if len(ocr_purestring) <= 5:
            yield "IMG_TEXT_NEARLY_EMPTY"
        else:
            ocr_flag = True
            yield "IMG_TEXT_[%s]_THAN_AVG" % ('GREATER' if len(ocr_purestring) >= kwargs.get('text_avg') else 'SMALLER')



    # ocr text information
    if ocr_flag:
        ocr_wordlist = span.text.lower().split('\n')
        ocr_wordlist = [w for w in ocr_wordlist if not w == '']
        doc_title = span.document.name
        title_count = reg_count = 0
        for w in ocr_wordlist:
            if fuzz.partial_ratio(w, doc_title) >= 70:
                title_count += 1
                yield "IMG_TEXT_MATCH_TITLE_{}".format(title_count)
            for p in prefix_list:
                if fuzz.partial_ratio(p, w) >= 70:
                    yield "IMG_TEXT_CONTAIN_PREFIX_{}".format(p.upper())
            for s in suffix_list:
                if fuzz.partial_ratio(s, w) >= 70:
                    yield "IMG_TEXT_CONTAIN_SUFFIX_{}".format(s.upper())
            if re.search(org_rgx, w):
                reg_count += 1
                yield "IMG_TEXT_REG_MATCH_{}".format(reg_count)


    # image description information
    reg_count = 0
    for w in span.description.split(" "):
        if re.search(org_rgx, w):
            reg_count += 1
            yield "IMG_DESC_REG_MATCH_{}".format(reg_count)

    for w in prefix_list:
        if not span.description.lower().find(w) == -1:
            yield "IMG_DESC_CONTAIN_PREFIX_{}".format(w.upper())

    for w in suffix_list:
        if not span.description.lower().find(w) == -1:
            yield "IMG_DESC_CONTAIN_SUFFIX_{}".format(w.upper())


    for w in keyword_list:
        if not span.description.lower().find(w) == -1:
            yield "IMG_DESC_CONTAIN_KEYWORD_{}".format(w.upper())
        if ocr_flag:
            if not span.text.lower().find(w) == -1:
                yield "IMG_TEXT_CONTAIN_KEYWORD_{}".format(w.upper())

    for w in black_word_list:
        if not span.description.lower().find(w) == -1:
            yield "IMG_DESC_CONTAIN_BLACKLISTED_WORD_{}".format(w.upper())
        # if not span.text.lower().find(w) == -1:
        #     yield "IMG_TEXT_CONTAIN_BLACKLISTED_WORD_{}".format(w.upper())


    if fuzz.partial_ratio(span.description, span.document.name) >= 50:
        yield "IMG_DESC_DOC_TITLE_MATCH"
    else:
        yield "IMG_DESC_DOC_TITLE_NOT_MATCH"


    ### Information from previous paragraph
    img = lf_helpers.find_image(span)
    prev_content = None
    for sibs in img.previous_siblings:
        if sibs:
            if sibs.has_attr('name') and sibs.name == 'br':
                continue
            if sibs.has_attr('get') and ('image_table' in sibs.get('class', [])):
                continue
            # print('find prev content!')
            prev_content = sibs.text
            break

    if prev_content and not span.name == '':
        pre,  post = lf_helpers.get_near_string(prev_content, span.name)
        if pre and post:
            for w in prefix_list:
                if not pre.lower().find(w) == -1:
                    yield "IMG_PREVPARA_PRE_CONTAIN_PREFIX_{}".format(w.upper())
                if not post.lower().find(w) == -1:
                    yield "IMG_PREVPARA_POST_CONTAIN_PREFIX_{}".format(w.upper())

            for w in suffix_list:
                if not pre.lower().find(w) == -1:
                    yield "IMG_PREVPARA_PRE_CONTAIN_SUFFIX_{}".format(w.upper())
                if not post.lower().find(w) == -1:
                    yield "IMG_PREVPARA_POST_CONTAIN_SUFFIX_{}".format(w.upper())


            for w in keyword_list:
                if not pre.lower().find(w) == -1:
                    yield "IMG_PREVPARA_PRE_CONTAIN_KEYWORD_{}".format(w.upper())
                if not post.lower().find(w) == -1:
                    yield "IMG_PREVPARA_POST_CONTAIN_KEYWORD_{}".format(w.upper())

