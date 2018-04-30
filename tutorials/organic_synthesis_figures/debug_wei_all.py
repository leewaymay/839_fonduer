from fonduer import SnorkelSession, candidate_subclass, Document
from fonduer import candidate_subclass
from .set_env import set_up
from scipy import sparse
from fonduer import HTMLPreprocessor, OmniParser
from fonduer.candidates import OmniDetailedFigures
from .product_spaces import OmniNgramsProd
from fonduer import CandidateExtractor
from fonduer import BatchLabelAnnotator
from .organic_lfs import *


PARALLEL = 1 # assuming a quad-core machine
session = SnorkelSession()
Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])
docs_path, pdf_path = set_up(name='wei')

restart = True
with_image_features = True
max_docs = 24
splits = 5 / 6

doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path,
                           blacklist=['style', 'script', 'meta', 'noscript'])

if restart:
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

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
org_rgx = get_rgx_matcher()
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

from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features

# Only need to do this once
print('Generating image features')
if restart:
    gen_image_features(docs_path=docs_path)

featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)
print('Generating other features')
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
F_test = featurizer.apply(split=1, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
print('Merging image features')


if with_image_features:
    F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0))  # concatenate dense with sparse matrix
    F_test = sparse.hstack(featurizer.load_matrix_and_image_features(split=1))  # concatenate dense with sparse matrix
else:
    F_train = featurizer.load_matrix(split=0)
    F_test = featurizer.load_matrix(split=1)

print(F_train.shape)
print(F_test.shape)

labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)

L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)

print(L_train.shape)

L_train.get_candidate(session, 0)

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
train_marginals = gen_model.marginals(L_train)

from fonduer import SparseLogisticRegression

disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)

test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]

train_score = disc_model.predictions(F_train)

for i, cand in enumerate(train_cands):
    print(cand.organic.text, '||||', cand.figure.url, train_score[i])

