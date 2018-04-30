from fonduer import Document, candidate_subclass, SnorkelSession
from scipy import sparse
from fonduer import HTMLPreprocessor, OmniParser
from fonduer.candidates import OmniDetailedFigures
from organic_spaces import OmniNgramsProd
from fonduer import CandidateExtractor
from fonduer import BatchLabelAnnotator
from organic_lfs import *
from regex_matcher import *
from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features
from fonduer import GenerativeModel
from fonduer import SparseLogisticRegression

Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])
session = SnorkelSession()
PARALLEL = 1

def parse(docs_path, pdf_path, max_docs):
    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
                               blacklist=['style', 'script', 'meta', 'noscript'])
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

def gen_cand():
    docs = session.query(Document).order_by(Document.name).all()
    ld   = len(docs)

    train_docs = set()
    test_docs  = set()
    data = [(doc.name, doc) for doc in docs]
    data.sort(key=lambda x: x[0])
    for i, (doc_name, doc) in enumerate(data):
        if i < splits * ld:
            train_docs.add(doc)
        else:
            test_docs.add(doc)
    from pprint import pprint
    pprint([x.name for x in train_docs])

    prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)
    fig_matcher = get_fig_matcher()

    figs = OmniDetailedFigures()
    prod_matcher = get_organic_matcher()

    candidate_extractor = CandidateExtractor(Org_Fig,
                            [prod_ngrams, figs],
                            [prod_matcher, fig_matcher],
                            candidate_filter=candidate_filter)

    candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
    candidate_extractor.apply(test_docs, split=1, parallelism=PARALLEL)
    train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
    test_cands = session.query(Org_Fig).filter(Org_Fig.split == 1).all()
    print("Number of train candidates: {}\nNumber of test candidates: {}".format(len(train_cands), len(test_cands)))
    return train_cands, test_cands

def gen_feature():
    featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
    print('Generating other features')
    F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
    F_test = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL) # generate sparse features
    # Only need to do this once
    print('Generating image features')
    return F_train, F_test

def gen_image_feature(docs_path):
    gen_image_features(docs_path=docs_path)

def load_feature(with_image_features=True):
    featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
    if with_image_features:
        print('Merging image features')
        F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0), format="csr")  # concatenate dense with sparse matrix
        F_test = sparse.hstack(featurizer.load_matrix_and_image_features(split=1), format="csr")  # concatenate dense with sparse matrix
    else:
        F_train = featurizer.load_matrix(split=0)
        F_test = featurizer.load_matrix(split=1)

    print(F_train.shape)
    print(F_test.shape)
    return F_train, F_test

def label_cand():

    labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)

    L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)

    print(L_train.shape)
    return L_train

def train_generative():
    gen_model = GenerativeModel()
    labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)
    L_train = labeler.load_matrix(split=0)
    gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001 / L_train.shape[0], reg_param=0)
    train_marginals = gen_model.marginals(L_train)
    return train_marginals

def train_discrimative(F_train, train_marginals):
    disc_model = SparseLogisticRegression()
    disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)
    return disc_model
