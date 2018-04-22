from fonduer.models import ImplicitSpan
from fonduer.snorkel.models import TemporarySpan
from fonduer.models import TemporaryImage, TemporaryDetailedImage
from fonduer.lf_helpers import *
from lxml.html import fromstring
from lxml import etree
from bs4 import BeautifulSoup
import re

FEAT_PRE = "COMBINED_"
DEF_VALUE = 1

binary_feats = {}
dfs_traversal = {}

def get_combined_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if len(args) != 2:
            raise NotImplementedError("Only handles binary candidates currently")

        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(args[0]))
        if not (isinstance(args[1], TemporaryDetailedImage)):
            raise ValueError("Accepts detailed_image-type arguments, %s-type found." % type(args[1]))

        span, fig = args

        if candidate.id not in binary_feats:
            binary_feats[candidate.id] = set()

        for f, v in structure_binary_features(span, fig):
            binary_feats[candidate.id].add((f, v))

        # for f, v in html_binary_features(span, fig):
        #     binary_feats[candidate.id].add((f, v))

        for f, v in context_binary_features(span, fig):
            binary_feats[candidate.id].add((f, v))

        for f, v in mention_distance(span, fig):
            binary_feats[candidate.id].add((f, v))

        for f, v in binary_feats[candidate.id]:
            yield candidate.id, FEAT_PRE + f, v


def structure_binary_features(organic, figure):
    if not organic.sentence.is_visual(): return
    for f, v in page_aligned(organic, figure):
        yield f, v
    for f, v in visual_distance(organic, figure):
        yield f, v

def html_binary_features(organic, figure):
    doc_text = figure.document.text
    soup = BeautifulSoup(doc_text, 'html.parser')
    image_node = find_image_in_html(figure, soup)
    if image_node == None:
        return
    #TODO use image node to produce more features
    #TODO use beautiful soup to find first appearance
    pass

def context_binary_features(organic, figure):
    for info in fig_contains_org(organic, figure, scores=[50, 75, 90, 100]):
        yield info, DEF_VALUE
    for info in org_contains_fig_name(organic, figure, scores=[75, 90, 100]):
        yield info, DEF_VALUE
    for info in fig_text_matches_org_text(organic, figure, scores=[25, 50, 75]):
        yield info, DEF_VALUE
    keywords = ['synthesis', 'reaction', 'produce', 'yield', 'formation', 'approach']
    if both_contain_keywords(organic, figure, keywords):
        yield "BOTH_CONTAIN_KEYWORD", DEF_VALUE
    for info in search_fig_first_apprearance(organic, figure):
        yield info, DEF_VALUE


def page_aligned(organic, figure):
    if is_same_org_fig_page(organic, figure):
        yield "SAME_PAGE", DEF_VALUE

    if is_same_sent_fig_page(organic, figure):
        yield "SENTENCE_SAME_PAGE", DEF_VALUE

    for num_page in range(1, 5):
        if is_nearby_org_fig_page(organic, figure, num_page):
            yield "WITHIN_{}_PAGE".format(num_page), DEF_VALUE

    if fig_on_prev_page(organic, figure):
        yield "FIG_PREV_PAGE", DEF_VALUE

    if org_on_prev_page(organic, figure):
        yield "ORG_PREV_PAGE", DEF_VALUE

def visual_distance(organic, figure):
    num_splits = 4
    distance_ratio = [(float)(i+1)/num_splits for i in range(num_splits)]

    for ratio in distance_ratio:
        for dist_info in within_distance(organic, figure, ratio):
            yield dist_info, DEF_VALUE

    for ahead_info in ahead_feature(organic, figure):
        yield ahead_info, DEF_VALUE


def find_image_in_html(figure, soup):
    for candidate in soup.find_all('div', class_='image_table'):
        if candidate.img.get('src') == figure.url:
            return candidate
    return None


def mention_distance(organic, figure):
    mention = ' '.join(filter(None, figure.name.split(' ')))  # trim extra spaces
    doc = figure.document
    root = fromstring(doc.text)
    tree = etree.ElementTree(root)
    if mention:
        if doc not in dfs_traversal:
            dfs_traversal[doc] = list([tree.getpath(it) for it in root.getiterator()])
        for phrase in doc.phrases:
            text = ' '.join(filter(None, phrase.text.split(' ')))  # trim extra spaces
            if mention in text:
                yield _get_depth_distance(organic.sentence.xpath, phrase.xpath), DEF_VALUE
                yield _get_iter_distance(doc, root, organic.sentence.xpath, phrase.xpath), DEF_VALUE


def _get_depth_distance(xpath1, xpath2):
    node1, node2 = xpath1.split('/'), xpath2.split('/')
    distance = len(node1) + len(node2)
    for n1, n2 in zip(node1, node2):
        if n1 == n2:
            distance -= 2
        else:
            break
    return "DEPTH_DISTANCE_[%s]" % distance


def _get_iter_distance(doc, root, xpath1, xpath2):
    id1, id2 = 0, 0
    for i, xpath in enumerate(dfs_traversal[doc]):
        pattern = xpath1 + '([d+])?'
        pattern = pattern.replace('[', '\[')
        pattern = pattern.replace(']', '\]')
        if re.match(pattern, xpath):
            id1 = i
        pattern = xpath2 + '([d+])?'
        pattern = pattern.replace('[', '\[')
        pattern = pattern.replace(']', '\]')
        if re.match(pattern, xpath):
            id2 = i
    cnt = 0
    for i in range(min(id1, id2) + 1, max(id1, id2)):
        if len(root.xpath(dfs_traversal[doc][i])[0]) == 0:
            cnt += 1

    thresholds = list(range(0, 1001, 50)) + [float('inf')]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= cnt < thresholds[i+1]:
            return "ITER_DISTANCE_[%s]" % thresholds[i]






