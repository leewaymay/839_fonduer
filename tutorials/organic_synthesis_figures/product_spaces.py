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
                 n_max=3,
                 expand=False,
                 split_tokens=' '):
        """:param parts_by_doc: a dictionary d where d[document_name.upper()] = [partA, partB, ...]"""
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=' ')
        self.parts_by_doc = parts_by_doc
        self.expander = lambda x: [x]

    def apply(self, session, context):
        for ts in OmniNgrams.apply(self, session, context):
            yield ts

