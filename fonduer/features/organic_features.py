from fonduer.models import ImplicitSpan
from fonduer.snorkel.models import TemporarySpan
import re

FEAT_PRE = "CORE_"
DEF_VALUE = 1

unary_feats = {}


def get_organic_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." %
                             type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]

            if span.stable_id not in unary_feats:
                unary_feats[span.stable_id] = set()
                for f in _generate_core_feats(span):
                    unary_feats[span.stable_id].add(f)

            for f in unary_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + f, DEF_VALUE

        # Binary candidates
        elif len(args) == 2:
            span, pre = args[0], "e1_"
            if span.stable_id not in unary_feats:
                unary_feats[span.stable_id] = set()
                for f in _generate_core_feats(span):
                    unary_feats[span.stable_id].add(f)
                for f in _generate_sentence_feats(span):
                    unary_feats[span.stable_id].add(f)
                for f in _generate_document_feats(span):
                    unary_feats[span.stable_id].add(f)
                for f in _generate_caption_feats(span):
                    unary_feats[span.stable_id].add(f)
                for f in _generate_image_feats(span):
                    unary_feats[span.stable_id].add(f)

            for f in unary_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + pre + f, DEF_VALUE
        else:
            raise NotImplementedError(
                "Only handles unary and binary candidates currently")


def _generate_core_feats(span):
    string = span.get_span()
    if string.upper() == string:
        yield "ABBREVIATION"

    if re.search('\d', string) is not None:
        yield "CONTAINS_DIGIT"

    if '-' in string:
        yield "CONTAINS_DASH"

    if ',' in string and string[-1] != ',' and string[0] != ',':
        yield "CONTAINS_COMMA"

    yield "DEP_LABELS_[{}]".format('_'.join(span.dep_labels).upper())
    yield "DEP_PARENTS_[{}]".format('_'.join(map(str, span.dep_parents)))
    yield "NER_TAGS_[{}]".format('_'.join(span.ner_tags).upper())
    yield "PAGE_{}".format(span.page)
    yield "POS_TAGS_[{}]".format('_'.join(span.pos_tags).upper())


def _generate_sentence_feats(span):
    sentence = span.sentence.text
    string = span.get_span()
    keywords = ['synthesis', 'syntheses', 'made', 'catalyze', 'generate', 'product', 'produce',
                'formation', 'developed', 'approach', 'yields']

    freq = sentence.count(string)
    if freq == 1:
        yield "ONCE_IN_SENTENCE".format(freq)
    if freq == 2:
        yield "TWICE_IN_SENTENCE".format(freq)
    if freq == 3:
        yield "THREE_TIMES_IN_SENTENCE".format(freq)
    if freq > 3:
        yield "MANY_TIMES_IN_SENTENCE".format(freq)
    for kw in keywords:
        if kw in sentence:
            yield "CONTAINS_{}".format(kw.upper())
        if span.char_start - (sentence.find(kw) + len(kw) + 1) < 5 or \
            sentence.find(kw) - span.char_end < 5:
            yield "NEAR_{}".format(kw.upper())
    if span.char_start - (sentence.find('of') + 2) == 1:
        yield "CONTAINS_(OF)"
    if span.char_start - (sentence.find('of the') + 7) == 1:
        yield "CONTAINS_(OF_THE)"

def _generate_document_feats(span):
    phrase_num = span.sentence.phrase_num
    doc = span.sentence.document
    string = span.get_span()
    if string in doc.name:
        yield "APPEARED_IN_DOC_NAME"
    freq = 0
    for i in range(len(doc.phrases)):
        if i != phrase_num and string in doc.phrases[i].words:
            freq += 1
        if 'summar' in doc.phrases[i].text or 'conclusion' in doc.phrases[i].text:
            yield "APPEARED_IN_CONCLUSION"
    if freq == 1:
        yield "ONCE_IN_DOCUMENT".format(freq)
    if freq == 2:
        yield "TWICE_IN_DOCUMENT".format(freq)
    if freq == 3:
        yield "THREE_TIMES_IN_DOCUMENT".format(freq)
    if freq > 3:
        yield "MANY_TIMES_IN_DOC".format(freq)

def _generate_caption_feats(span):
    doc = span.sentence.document
    string = span.get_span()
    freq = 1
    for i in range(len(doc.detailed_images)):
        freq += doc.detailed_images[i].description.count(string)
    if freq == 1:
        yield "ONCE_IN_CAPTIONS".format(freq)
    if freq == 2:
        yield "TWICE_IN_CAPTIONS".format(freq)
    if freq == 3:
        yield "THREE_TIMES_IN_CAPTIONS".format(freq)
    if freq > 3:
        yield "MANY_TIMES_IN_CAPTIONS".format(freq)

def _generate_image_feats(span):
    doc = span.sentence.document
    string = span.get_span()
    freq = 1
    for i in range(len(doc.detailed_images)):
        freq += doc.detailed_images[i].text.count(string)
    if freq == 1:
        yield "ONCE_IN_IMAGES".format(freq)
    if freq == 2:
        yield "TWICE_IN_IMAGES".format(freq)
    if freq == 3:
        yield "THREE_TIMES_IN_IMAGES".format(freq)
    if freq > 3:
        yield "MANY_TIMES_IN_IMAGES".format(freq)