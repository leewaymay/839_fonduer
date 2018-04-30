PARALLEL = 1

import os
ATTRIBUTE = "organic_figure"
os.environ['FONDUERHOME'] = '/Users/Zitman/Documents/Graduate/Courses/CS839/Project/839_fonduer'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']
docs_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/data/pdf/'

from scipy import sparse
from fonduer import SnorkelSession, candidate_subclass 
from fonduer import HTMLPreprocessor, OmniParser


session = SnorkelSession()
Org_Fig = candidate_subclass('Org_Fig', ['organic','figure'])


# ## Parse the documents


max_docs = 24
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)
corpus_parser = OmniParser(structural=True, lingual=True, visual=True, 
                           pdf_path=pdf_path,
                           blacklist=['style', 'script', 'meta', 'noscript'])


# Run this to get the document parsed


corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)


# ## Split the documents



from fonduer import Document

docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)


# In[19]:


train_docs = set()
test_docs  = set()
splits = 5 / 6
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits * ld:
        train_docs.add(doc)
    else:
        test_docs.add(doc)
print([x.name for x in train_docs])


# In[21]:


from fonduer.snorkel.matchers import LambdaFunctionMatcher, Intersect, Union
from fonduer.snorkel.matchers import RegexMatchSpan
from regex_matcher import get_rgx_matcher


# In[22]:


org_rgx = get_rgx_matcher()

rgx_matcher = RegexMatchSpan(rgx=org_rgx, longest_match_only=True, ignore_case=False)
blacklist = ['CAS', 'PDF', 'RSC', 'SAR', 'TEM']
prod_blacklist_lambda_matcher = LambdaFunctionMatcher(func=lambda x: x.text not in blacklist, ignore_case=False)
blacklist_rgx = ['methods?.?']
prod_blacklist_rgx_lambda_matcher = LambdaFunctionMatcher(
    func=lambda x: all([re.match(r, x.text) is None for r in blacklist_rgx]), ignore_case=False)

#prod_matcher = rgx_matcher
prod_matcher = Intersect(rgx_matcher, prod_blacklist_lambda_matcher, prod_blacklist_rgx_lambda_matcher)


# In[23]:


from fonduer import CandidateExtractor
from fonduer.lf_helpers import *
import re

def candidate_filter(c):
    (organic, figure) = c
    if same_file(organic, figure):
        if mentionsFig(organic, figure) or mentionsOrg(figure, organic):
            return True


from organic_spaces import OmniNgramsProd
prod_ngrams = OmniNgramsProd(parts_by_doc=None, n_max=3)

from fonduer.matchers import LambdaFunctionFigureMatcher

def white_black_list_matcher(fig):
    white_list = ['synthesis', 'plausible']
    black_list = ['spectra', 'x-ray', 'copyright', 'structur', 'application']

    fig_desc = fig.figure.description.lower()
    in_white = in_black = False
    if any(fig_desc.find(v) >= 0 for v in white_list): in_white = True
    if any(fig_desc.find(v) >= 0 for v in black_list): in_black = True
    if in_black and (not in_white):
        return False
    return True

def contain_organic_matcher(fig):
    # filter 2
    desc_wordlist = fig.figure.description.lower().split(' ')
    if any(re.search(org_rgx, w) for w in desc_wordlist): return True
    if not fig.figure.text == '':
        orc_wordlist = fig.figure.text.lower().split('\n')
        orc_wordlist = [w for w in orc_wordlist if not w == '']
        if any(re.search(org_rgx, w) for w in orc_wordlist): return True
    return False

fig_matcher1 = LambdaFunctionFigureMatcher(func=white_black_list_matcher)
fig_matcher2 = LambdaFunctionFigureMatcher(func=contain_organic_matcher)
fig_matcher = Union(fig_matcher1, fig_matcher2)


# In[ ]:


from fonduer.candidates import OmniDetailedFigures

figs = OmniDetailedFigures()

candidate_extractor = CandidateExtractor(Org_Fig,
                        [prod_ngrams, figs],
                        [prod_matcher, fig_matcher],
                        candidate_filter=candidate_filter)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)
candidate_extractor.apply(test_docs, split=1, parallelism=PARALLEL)


# ## Run this to get the training and testing candidates

# In[24]:


train_cands = session.query(Org_Fig).filter(Org_Fig.split == 0).all()
test_cands = session.query(Org_Fig).filter(Org_Fig.split == 1).all()
print("Number of train candidates: {}\nNumber of test candidates: {}".format(len(train_cands), len(test_cands)))


# ## Generate features

# In[25]:


from fonduer import BatchFeatureAnnotator
from fonduer.features.features import get_organic_image_feats
from fonduer.features.read_images import gen_image_features


# In[ ]:


# Only need to do this once
print('Generating image features')
# session.execute("delete from context where stable_id like '%feature%'")
gen_image_features(docs_path=docs_path)


# In[28]:


featurizer = BatchFeatureAnnotator(Org_Fig, f=get_organic_image_feats)


# ### Run this to re-generate the features

# In[ ]:


print('Generating other features')
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL) # generate sparse features
F_test = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL) # generate sparse features
print('Merging image features')


# ### Run this to reload the features

# In[29]:


F_train = sparse.hstack(featurizer.load_matrix_and_image_features(split=0)).toarray()  # concatenate dense with sparse matrix
F_test = sparse.hstack(featurizer.load_matrix_and_image_features(split=1), format="csr").toarray()  # concatenate dense with sparse matrix


# In[26]:


from fonduer import BatchLabelAnnotator
from organic_lfs import *


# put more labeling functions in ```organic_lfs```

# ## Add more labeling functions here

# In[34]:


org_fig_lfs = [
    LF_fig_name_match,
    LF_text_desc_match,
    LF_ocr_text_match,
    LF_text_length_match,
    LF_match_whitelist,
    LF_match_blacklist,
    LF_match_page,
    LF_pos_near,
    LF_organic_compound,
    LF_synthesis_of,
    LF_product_of,
    LF_first_period,
]


# In[35]:


labeler = BatchLabelAnnotator(Org_Fig, lfs=org_fig_lfs)


# In[ ]:


L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)


# In[36]:


L_train = labeler.load_matrix(split=0)


# In[37]:


L_train.shape


# In[38]:


L_train.get_candidate(session, 0)


# In[46]:


from fonduer import GenerativeModel


# In[47]:


gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
train_marginals = gen_model.marginals(L_train)


# In[50]:


print(gen_model.weights.lf_accuracy)


# In[52]:


L_test = labeler.apply_existing(split = 1)


# In[30]:


F_train.shape


# In[31]:


F_test.shape


# In[ ]:


from fonduer import LogisticRegression

disc_model = LogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)
test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]
train_score = disc_model.predictions(F_train)


# In[32]:


F_train_sparse = featurizer.load_matrix(split = 0)


# In[55]:


F_test_sparse = featurizer.load_matrix(split = 1)


# In[56]:


F_test_sparse.get_candidate(session,0)


# In[39]:


# load gold label
from tutorials.organic_synthesis_figures.organic_utils import load_organic_labels


# In[41]:


gold_file = os.environ['FONDUERHOME'] + '/tutorials/organic_synthesis_figures/organic_gold.csv'

load_organic_labels(session, Org_Fig, gold_file, ATTRIBUTE ,
                    annotator_name='gold')


# In[42]:


from fonduer import load_gold_labels


# In[43]:


L_gold_train = load_gold_labels(session, annotator_name="gold", split=0)
print(L_train.lf_stats(L_gold_train))


# In[44]:


L_gold_test = load_gold_labels(session, annotator_name="gold", split=1)


# In[53]:


prec, rec, f1 = gen_model.score(L_test, L_gold_test)


# In[54]:


print("precision ", prec, " recall ", rec, " f1 ", f1)

