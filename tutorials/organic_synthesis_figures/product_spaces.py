from __future__ import print_function
from builtins import chr
from builtins import str
from builtins import range
from difflib import SequenceMatcher
import re
from fonduer.candidates import OmniNgrams
from fonduer.models import TemporaryImplicitSpan


class OmniNgramsProd(OmniNgrams):
    def __init__(self,
                 parts_by_doc=None,
                 n_max=1,
                 expand=False,
                 split_tokens=' '):
        """:param parts_by_doc: a dictionary d where d[document_name.upper()] = [partA, partB, ...]"""
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=' ')
        self.parts_by_doc = parts_by_doc
        self.expander = lambda x: [x]

    def apply(self, session, context):
        for ts in OmniNgrams.apply(self, session, context):
            value = ts.get_span()
            yield TemporaryImplicitSpan(
                sentence=ts.sentence,
                char_start=ts.char_start,
                char_end=ts.char_end,
                expander_key=u'prod_expander',
                position=0,
                text=value,
                words=[value],
                lemmas=[value],
                pos_tags=[ts.get_attrib_tokens('pos_tags')[-1]],
                ner_tags=[ts.get_attrib_tokens('ner_tags')[-1]],
                dep_parents=[ts.get_attrib_tokens('dep_parents')[-1]],
                dep_labels=[ts.get_attrib_tokens('dep_labels')[-1]],
                page=[ts.get_attrib_tokens('page')[-1]]
                if ts.sentence.is_visual() else [None],
                top=[ts.get_attrib_tokens('top')[-1]]
                if ts.sentence.is_visual() else [None],
                left=[ts.get_attrib_tokens('left')[-1]]
                if ts.sentence.is_visual() else [None],
                bottom=[ts.get_attrib_tokens('bottom')[-1]]
                if ts.sentence.is_visual() else [None],
                right=[ts.get_attrib_tokens('right')[-1]]
                if ts.sentence.is_visual() else [None],
                meta=None)

