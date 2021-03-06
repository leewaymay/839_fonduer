
# coding: utf-8

# # Tutorial: Extracting Maximum Storage Temperatures for Transistors from PDF Datasheets

# # Introduction
# 
# We will walk through the process of using `Fonduer` to extract relations from [**richly formatted** data](https://hazyresearch.github.io/snorkel/blog/fonduer.html), where information is conveyed via combinations of textual, structural, tabular, and visual expressions, as seen in webpages, business reports, product specifications, and scientific literature.
# 
# In this tutorial, we use `Fonduer` to identify mentions of the maximum storage temperature of transistors (e.g. `150°C`) in a corpus of transistor datasheets from [Digikey.com](https://www.digikey.com/products/en/discrete-semiconductor-products/transistors-bipolar-bjt-single/276).
# 
# The tutorial is broken into several parts, each covering a Phase of the `Fonduer` pipeline (as outlined in the [paper](https://arxiv.org/abs/1703.05028)), and the iterative KBC process:
# 
# 1. KBC Initialization
# 2. Candidate Generation and Multimodal Featurization
# 3. Probabilistic Relation Classification
# 4. Error Analysis and Iterative KBC
# 
# In addition, we show how users can iteratively improve labeling functions to improve relation extraction quality.
# 
# # Phase 1: KBC Initialization
# 
# In this first phase of `Fonduer`'s pipeline, `Fonduer` uses a user specified _schema_ to initialize a relational database where the output KB will be stored. Furthermore, `Fonduer` iterates over its input _corpus_ and transforms each document into a unified data model, which captures the variability and multimodality of richly formatted data. This unified data model then servers as an intermediate representation used in the rest of the phases.
# 
# This preprocessed data is saved to a database. Connection strings can be specified by setting the `SNORKELDB` environment variable. If no database is specified, then SQLite at `./snorkel.db` is created by default. However, to enabled parallel execution, we use PostgreSQL throughout this tutorial.
# 
# We initialize several variables for convenience that define what the database should be called and what level of parallelization the `Fonduer` pipeline will be run with. In the code below, we use PostgreSQL as our database backend. 
# 
# Before you continue, please make sure that you have PostgreSQL installed and have created a new database named `stg_temp_max`.

# In[1]:



import os
import sys

PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "stg_temp_max"
os.environ['FONDUERHOME'] = '/Users/Zitman/Documents/Graduate/Courses/CS839/Project/fonduer-master'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']


# ## 1.1 Defining a Candidate Schema
# 
# We first initialize a `SnorkelSession`, which manages the connection to the database automatically, and enables us to save intermediate results.

# In[2]:


from fonduer import SnorkelSession

session = SnorkelSession()


# Next, we define the _schema_ of the relation we want to extract. This must be a subclass of Candidate, and we define it using a helper function. Here, we define a binary relation which connects two Span objects of text. This is what creates the relation's database table if it does not already exist.

# In[3]:


from fonduer import candidate_subclass

Part_Attr = candidate_subclass('Part_Attr', ['part','attr'])


# ## 1.2 Parsing and Transforming the Input Documents into Unified Data Models
# 
# Next, we load the corpus of datasheets and transform them into the unified data model. Each datasheet has a PDF and HTML representation. Both representations are used in conjunction to create a robust unified data model with textual, structural, tabular, and visual modality information. Note that since each document is independent of each other, we can parse the documents in parallel. Note that parallel execution will not work with SQLite, the default database engine. We depend on PostgreSQL for this functionality.
# 
# ### Configuring an `HTMLPreprocessor`
# We start by setting the paths to where our documents are stored, and defining a `HTMLPreprocessor` to read in the documents found in the specified paths. `max_docs` specified the number of documents to parse. For the sake of this tutorial, we only look at 100 documents.
# 
# **Note that you need to have run `download_data.sh` before executing these next steps or you won't have the documents needed for the tutorial.**

# In[4]:


from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/html/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/pdf/'

max_docs = 10
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)


# ### Configuring an `OmniParser`
# Next, we configure an `OmniParser`, which serves as our `CorpusParser` for PDF documents. We use [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) as a preprocessing tool to split our documents into phrases and tokens, and to provide annotations such as part-of-speech tags and dependency parse structures for these phrases. In addition, we can specify which modality information to include in the unified data model for each document. Below, we enable all modality information.

# In[5]:


corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
corpus_parser.apply(doc_preprocessor, parallelism=1)


# We can then use simple database queries (written in the syntax of [SQLAlchemy](http://www.sqlalchemy.org/), which `Fonduer` uses) to check how many documents and phrases (sentences) were parsed, or even check how many phrases and tables are contained in each document.

# In[26]:


from fonduer import Document, Phrase

print("Documents:", session.query(Document).count())
print("Phrases:", session.query(Phrase).count())


# ## 1.3 Dividing the Corpus into Test and Train
# 
# We'll split the documents 80/10/10 into train/dev/test splits. Note that here we do this in a non-random order to preverse the consistency in the tutorial, and we reference the splits by 0/1/2 respectively.

# In[28]:


docs = session.query(Document).order_by(Document.name).all()
ld   = len(docs)

train_docs = set()
dev_docs   = set()
test_docs  = set()
splits = (0.8, 0.9)
data = [(doc.name, doc) for doc in docs]
data.sort(key=lambda x: x[0])
for i, (doc_name, doc) in enumerate(data):
    if i < splits[0] * ld:
        train_docs.add(doc)
    elif i < splits[1] * ld:
        dev_docs.add(doc)
    else:
        test_docs.add(doc)
from pprint import pprint
pprint([x.name for x in train_docs])


# # Phase 2: Candidate Extraction & Multimodal Featurization
# Given the unified data model from Phase 1, `Fonduer` extracts relation candidates based on user-provided **matchers** and **throttlers**. Then, `Fonduer` leverages the multimodality information captured in the unified data model to provide multimodal features for each candidate.
# 
# ## 2.1 Candidate Extraction
# 
# The next step is to extract **candidates** from our corpus. A `candidate` is the object for which we want to make predictions. In this case, the candidates are pairs of transistor part numbers and their corresponding maximum storage temperatures as found in their datasheets. Our task is to predict which pairs are true in the associated document.
# 
# To do so, we write **matchers** to define which spans of text in the corpus are instances of each entity. Matchers can leverage a variety of information from regular expressions, to dictionaries, to user-defined functions. Furthermore, different techniques can be combined to form higher quality matchers. In general, matchers should seek to be as precise as possible while maintaining complete recall.
# 
# In our case, we need to write a matcher that defines a transistor part number and a matcher to define a valid temperature value.
# 
# ### Writing a simple temperature matcher
# 
# Our maximum storage temperature matcher can be a very simple regular expression since we know that we are looking for integers, and by inspecting a portion of our corpus, we see that maximum storage temperatures fall within a fairly narrow range.

# In[9]:


from fonduer import RegexMatchSpan, DictionaryMatch, LambdaFunctionMatcher, Intersect, Union

attr_matcher = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)


# ### Writing an advanced transistor part matcher
# 
# In contrast, transistor part numbers are complex expressions. Here, we show how transistor part numbers can leverage [naming conventions](https://en.wikipedia.org/wiki/Transistor#Part_numbering_standards.2Fspecifications) as regular expressions, and use a dictionary of known part numbers, and use user-defined functions together. First, we create a regular expression matcher for standard transistor naming conventions.

# In[10]:


### Transistor Naming Conventions as Regular Expressions ###
eeca_rgx = r'([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = r'(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = r'(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
others_rgx = r'((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT|SMMBT|PZT|FZT|STD|BUV|PBSS|KSC|CXT|FCX|CMPT){1}[\d]{2,4}[A-Z]{0,5}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)'

part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)


# Next, we can create a matcher from a dictionary of known part numbers:

# In[11]:


import csv

def get_digikey_parts_set(path):
    """
    Reads in the digikey part dictionary and yeilds each part.
    """
    all_parts = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            (part, url) = line
            all_parts.add(part)
    return all_parts

### Dictionary of known transistor parts ###
dict_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/digikey_part_dictionary.csv'
part_dict_matcher = DictionaryMatch(d=get_digikey_parts_set(dict_path))


# We can also use user-defined functions to further improve our matchers. For example, here we use patterns in the document filenames as a signal for whether a span of text in a document is a valid transistor part number.

# In[12]:


from builtins import range

def common_prefix_length_diff(str1, str2):
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            return min(len(str1), len(str2)) - i
    return 0

def part_file_name_conditions(attr):
    file_name = attr.sentence.document.name
    if len(file_name.split('_')) != 2: return False
    if attr.get_span()[0] == '-': return False
    name = attr.get_span().replace('-', '')
    return any(char.isdigit() for char in name) and any(char.isalpha() for char in name) and common_prefix_length_diff(file_name.split('_')[1], name) <= 2

add_rgx = '^[A-Z0-9\-]{5,15}$'

part_file_name_lambda_matcher = LambdaFunctionMatcher(func=part_file_name_conditions)
part_file_name_matcher = Intersect(RegexMatchSpan(rgx=add_rgx, longest_match_only=True), part_file_name_lambda_matcher)


# Then, we can union all of these matchers together to form our final part matcher.

# In[13]:


part_matcher = Union(part_rgx_matcher, part_dict_matcher, part_file_name_matcher)


# These two matchers define each entity in our relation schema.

# ### Define a relation's `ContextSpaces`
# 
# Next, in order to define the "space" of all candidates that are even considered from the document, we need to define a `ContextSpace` for each component of the relation we wish to extract.
# 
# In the case of transistor part numbers, the `ContextSpace` can be quite complex due to the need to handle implicit part numbers that are implied in text like "BC546A/B/C...BC548A/B/C", which refers to 9 unique part numbers. In addition, to handle these, we consider all n-grams up to 3 words long.
# 
# In contrast, the `ContextSpace` for temperature values is simpler: we only need to process different unicode representations of a (`-`), and don't need to look at more than two works at a time.
# 
# When no special preproessing like this is needed, we could have used the default `OmniNgrams` class provided by `snorkel.candidates`. For example, if we were looking to match polarities, which only take the form of "NPN" or "PNP", we could've used `attr_ngrams = OmniNgrams(n_max=1)`.

# In[14]:


from hardware_spaces import OmniNgramsPart, OmniNgramsTemp
    
part_ngrams = OmniNgramsPart(parts_by_doc=None, n_max=3)
attr_ngrams = OmniNgramsTemp(n_max=2)


# ### Defining candidate `Throttlers`
# 
# Next, we need to define **throttlers**, which allow us to further prune excess candidates and avoid unnecessarily materializing invalid candidates. Trottlers, like matchers, act as hard filters, and should be created to have high precision while maintaining complete recall, if possible.
# 
# Here, we create a throttler that discards candidates if they are in the same table, but the part and storage temperature are not vertically or horizontally aligned.

# In[15]:


from fonduer.lf_helpers import *
import re

def stg_temp_filter(c):
    (part, attr) = c
    if same_table((part, attr)):
        return (is_horz_aligned((part, attr)) or is_vert_aligned((part, attr)))
    return True

candidate_filter = stg_temp_filter


# ### Running the `CandidateExtractor`
# 
# Now, we have all the component necessary to perform candidate extraction. We have defined the "space" of things to consider for each candidate, provided matchers that signal when a valid mention is seen, and a throttler to prunes away excess candidates. We now can define the `CandidateExtractor` with the contexts to extract from, the matchers, and the throttler to use. 

# In[16]:


from fonduer import CandidateExtractor


candidate_extractor = CandidateExtractor(Part_Attr, 
                        [part_ngrams, attr_ngrams], 
                        [part_matcher, attr_matcher], 
                        candidate_filter=candidate_filter)

candidate_extractor.apply(train_docs, split=0, parallelism=PARALLEL)


# Here we specified that these `Candidates` belong to the training set by specifying `split=0`; recall that we're referring to train/dev/test as splits 0/1/2.

# In[17]:


train_cands = session.query(Part_Attr).filter(Part_Attr.split == 0).all()
print("Number of candidates:", len(train_cands))


# ### Repeating for development and test splits
# Finally, we rerun the same operation for the other two document divisions: dev and test. For each, we simply load the `Corpus` object and run them through the `CandidateExtractor`.

# In[18]:


for i, docs in enumerate([dev_docs, test_docs]):
    candidate_extractor.apply(docs, split=i+1)
    print("Number of candidates:", session.query(Part_Attr).filter(Part_Attr.split == i+1).count())


# In[27]:


session.query(Part_Attr).limit(10).all()


# ## 2.2 Multimodal Featurization
# Unlike dealing with plain unstructured text, `Fonduer` deals with richly formatted data, and consequently featurizes each candidate with a baseline library of multimodal features. 
# 
# ### Featurize with `Fonduer`'s optimized Postgres Feature Annotator
# We now annotate the candidates in our training, dev, and test sets with features. The `BatchFeatureAnnotator` provided by `Fonduer` allows this to be done in parallel to improve performance.
# 
# `featurizer.apply` takes three important arguments.
# * `split` defines which candidate set wer are dealing with. For example, `split=0` is the training set.
# * `replace_key_set` determine whether or not replace, or reinitialize, the set of features to apply to candidates. That is, when `replace_key_set` is true, key set of features will be replaced with the key set of the features found in the split that is being processed.
# * `parallelism` determines how many processes to run in parallel.
# 
# Notices that `replace_key_set=True` only for the first call to `featurizer.apply`, while the other calls have this parameter set to `False`. This is because we want to have the set of features we label candidates with defined by the features found in the set of training documents only. If a later call to `featurizer.apply` replaced the key set, then only the features of that particular split would be considered later in the pipeline.

# In[23]:


from fonduer import BatchFeatureAnnotator

featurizer = BatchFeatureAnnotator(Part_Attr)
F_train = featurizer.apply(split=0, replace_key_set=True, parallelism=PARALLEL)
print(F_train.shape)
F_dev = featurizer.apply(split=1, replace_key_set=False, parallelism=PARALLEL)
print(F_dev.shape)
F_test = featurizer.apply(split=2, replace_key_set=False, parallelism=PARALLEL)
print(F_test.shape)


# At the end of this phase, `Fonduer` has generated the set of candidates and the feature matrix. Note that Phase 1 and 2 are relatively static and typically are only executed once during the KBC process.
# 
# # Phase 3: Probabilistic Relation Classification
# In this phase, `Fonduer` applies user-defined **labeling functions**, which express various heuristics, patterns, and [weak supervision](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html) strategies to label our data, to each of the candidates to create a label matrix that is used by our data programming engine.
# 
# In the wild, hand-labeled training data is rare and expensive. A common scenario is to have access to tons of unlabeled training data, and have some idea of how to label them programmatically. For example:
# * We may be able to think of text patterns that would indicate a part and polarity mention are related, for example the word "temperature" appearing between them.
# * We may have access to an external knowledge base that lists some pairs of parts and polarities, and can use these to noisily label some of our mention pairs.
# Our labeling functions will capture these types of strategies. We know that these labeling functions will not be perfect, and some may be quite low-quality, so we will model their accuracies with a generative model, which `Fonduer` will help us easily apply.
# 
# Using data programming, we can then train machine learning models to learn which features are the most important in classifying candidates.
# 
# ### Loading Gold Data
# For convenience in error analysis and evaluation, we have already annotated the dev and test set for this tutorial, and we'll now load it using an externally-defined helper function.
# 
# Loading and saving external "gold" labels can be a bit messy, but is often a critical part of development, especially when gold labels are expensive and/or time-consuming to obtain. Snorkel stores all labels that are manually annotated in a **stable** format (called StableLabels), which is somewhat independent from the rest of Snorkel's data model, does not get deleted when you delete the candidates, corpus, or any other objects, and can be recovered even if the rest of the data changes or is deleted.
# 
# Our general procedure with external labels is to load them into the `StableLabel` table, then use `Fonduer`'s helpers to load them into the main data model from there. If interested in example implementation details, please see the script we now load:

# In[19]:


from hardware_utils import load_hardware_labels

gold_file = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/hardware_tutorial_gold.csv'
load_hardware_labels(session, Part_Attr, gold_file, ATTRIBUTE ,annotator_name='gold')


# ### Creating Labeling Functions
# 
# In `Fonduer`, our primary interface through which we provide training signal to the end extraction model we are training is by writing labeling functions (**LFs**) (as opposed to hand-labeling massive training sets).
# 
# A labeling function isn't anything special. It's just a Python function that accepts a `Candidate` as the input argument and returns `1` if it says the Candidate should be marked as true, `-1` if it says the `Candidate` should be marked as false, and `0` if it doesn't know how to vote and abstains. In practice, many labeling functions are unipolar: it labels only 1s and 0s, or it labels only -1s and 0s.
# 
# Recall that our goal is ultimately to train a high-performance classification model that predicts which of our Candidates are true mentions of maximum storage temperature relations. It turns out that we can do this by writing potentially low-quality labeling functions!
# 
# With `Fonduer`, labeling functions can be written using intuitive patterns discovered by inspecting the target corpus. A library of labeling function helpers can be found in `fonduer.lf_helpers`. 
# 
# For example, inspecting several document may reveal that storage temperatures are typically listed inside a table where the row header contains the word "storage". This intuitive pattern can be directly expressed as a labeling function. Similarly, the word "temperature" is an obvious positive signal.

# In[20]:


from fonduer.lf_helpers import *
import re

def LF_storage_row(c):
    return 1 if 'storage' in get_row_ngrams(c.attr) else 0

def LF_temperature_row(c):
    return 1 if 'temperature' in get_row_ngrams(c.attr) else 0


# We express several of these simple patterns below as a set of labeling functions:

# In[21]:


def LF_operating_row(c):
    return 1 if 'operating' in get_row_ngrams(c.attr) else 0

def LF_tstg_row(c):
    return 1 if overlap(
        ['tstg','stg','ts'], 
        list(get_row_ngrams(c.attr))) else 0


def LF_to_left(c):
    return 1 if 'to' in get_left_ngrams(c.attr, window=2) else 0

def LF_negative_number_left(c):
    return 1 if any([re.match(r'-\s*\d+', ngram) for ngram in get_left_ngrams(c.attr, window=4)]) else 0


# Then, we collect all of the labeling function we would like to use into a single list, which is provided as input to the `LabelAnnotator`.

# In[22]:


stg_temp_lfs = [
    LF_storage_row,
    LF_operating_row,
    LF_temperature_row,
    LF_tstg_row,
    LF_to_left,
    LF_negative_number_left
]


# ### Applying the Labeling Functions
# 
# Next, we need to actually run the LFs over all of our training candidates, producing a set of `Labels` and `LabelKeys` (just the names of the LFs) in the database. We'll do this using the `LabelAnnotator` class, a `UDF` which we will again run with `UDFRunner`. Note that this will delete any existing `Labels` and `LabelKeys` for this candidate set. Also note that we are using `Fonduer`'s optimized batch label annotator, which runs in parallel and depends on having Postgres as the backend database. 
# 
# By default, `labeler.apply` will drop the existing table of labeling functions and the label values for each candidate. However, this behavior can be controlled by three parameters to the function to imperove iteration performance and reduce redundant computation:
# - `split` defines which set to operate on (e.g. train, dev, or test)
# - `clear` can be `True` or `False`, and is `True` by default. When set to `False`, the labeling functioni table is not dropped, and the behavior of `labeler.apply` is defined by the following two parameters.
# - `update_keys` can be `True` or `False`. When `True`, the keys (which are each labeling function) are updated according to the set of labeling functions provided to the function. This should be set to `True` if new labeling functions are added. When `False`, no new LFs are evaluated and the keys of existing LFs remain the same.
# - `update_values` can be `True` or `False`. This defines how to resolve conflicts. When `True`, the values assigned to each candiate is updated to the new values when in conflict. This should be set to `True` if labeling function logic is edited, even though the name of the labeling function remains the same. When `False`, the existing labels assigned to each candidate are used, and newly computed labels are ignored.
# - `parallelism` is the amount of parallelism to use when labeling.
# 
# With this in mind, we set `clear=True` when we first apply our labeling functions, and this ensures that the table is created and intialized with proper keys and values.
# 
# In future iterations, we would typically set `clear=False, update_keys=True, update_values=True` so that we can simply update the set of LFs and their values without recreating the entire table. We will see how this is used later in the tutorial.

# In[23]:


from fonduer import BatchLabelAnnotator

labeler = BatchLabelAnnotator(Part_Attr, lfs = stg_temp_lfs)
L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
print(L_train.shape)


# Note that the returned matrix is a special subclass of the scipy.sparse.csr_matrix class, with some special features which we demonstrate below:

# In[24]:


L_train.get_candidate(session, 0)


# We can also view statistics about the resulting label matrix.
# * **Coverage** is the fraction of candidates that the labeling function emits a non-zero label for.
# * **Overlap** is the fraction candidates that the labeling function emits a non-zero label for and that another labeling function emits a non-zero label for.
# * **Conflict** is the fraction candidates that the labeling function emits a non-zero label for and that another labeling function emits a conflicting non-zero label for.
# 
# In addition, because we have already loaded the gold labels, we can view the emperical accuracy of these labeling functions when compared to our gold labels:

# In[25]:


from fonduer import load_gold_labels
L_gold_train = load_gold_labels(session, annotator_name='gold', split=0)
L_train.lf_stats(L_gold_train)


# ### Fitting the Generative Model
# 
# Now, we'll train a model of the LFs to estimate their accuracies. Once the model is trained, we can combine the outputs of the LFs into a single, noise-aware training label set for our extractor. Intuitively, we'll model the LFs by observing how they overlap and conflict with each other.

# In[26]:


from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)


# We now apply the generative model to the training candidates to get the noise-aware training label set. We'll refer to these as the training marginals:

# In[27]:


train_marginals = gen_model.marginals(L_train)


# We'll look at the distribution of the training marginals:

# In[28]:


import matplotlib.pyplot as plt
plt.hist(train_marginals, bins=20)
plt.show()


# We can view the learned accuracy parameters as well.

# In[29]:


gen_model.weights.lf_accuracy


# ### Using the Model to Iterate on Labeling Functions
# 
# Now that we have learned the generative model, we can stop here and use this to potentially debug and/or improve our labeling function set. First, we apply the LFs to our development set:

# In[30]:


L_dev = labeler.apply_existing(split=1)


# In[31]:


L_dev.shape


# Then, we get the score of the generative model:

# In[32]:


from fonduer import load_gold_labels
L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
prec, rec, f1 = gen_model.score(L_dev, L_gold_dev)


# We can also view statistics about the labeling function's learned accuracy and compare them to the emperical accuracy.

# In[33]:


L_dev.lf_stats(L_gold_dev, gen_model.weights.lf_accuracy)


# ### Interpreting Generative Model Performance
# 
# At this point, we should be getting an F1 score of around 0.6 to 0.7 on the development set, which is pretty good! However, we should be very careful in interpreting this. Since we developed our labeling functions using this development set as a guide, and our generative model is composed of these labeling functions, we expect it to score very well here!
# 
# In fact, it is probably somewhat overfit to this set. However this is fine, since in the next, we'll train a more powerful end extraction model which will generalize beyond the development set, and which we will evaluate on a blind test set (i.e. one we never looked at during development).
# 
# 
# ### Training the Discriminative Model
# 
# Now, we'll use the noisy training labels we generated in the last part to train our end extraction model. For this tutorial, we will be training a simple - but fairly effective - logistic regression model.
# 
# We use the training marginals to train a discriminative model that classifies each Candidate as a true or false mention. 

# In[34]:


from fonduer import SparseLogisticRegression

disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)


# ### Evaluating on the Test Set
# In this final section, we'll get the score we've been after: the performance of the extraction model on the blind test set (split 2). First, we load the test set labels and gold candidates from earlier:

# In[35]:


from fonduer import load_gold_labels
L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)


# Now, we score using the discriminitive model:

# In[36]:


test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]


# In[37]:


from hardware_utils import entity_level_f1

import pickle
pickle_file = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/parts_by_doc_dict.pkl'
with open(pickle_file, 'rb') as f:
    parts_by_doc = pickle.load(f)

(TP, FP, FN) = entity_level_f1(true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc)


# # Phase 4:  Error Analysis & Iterative KBC
# 
# During the development process, we can iteratively improve the quality of our labeling functions through error analysis, without executing the full pipeline as in previous techniques. 
# 
# You may have noticed that our final score is about 50 F1 points. To remedy this and improve our quality, we can perform error analysis to understand what kinds of patterns we may have missed, or what issues exist with our labeling functions. Then, we can edit our set of labeling functions and rerun Phase 3, Probabilistic Relation Classification. 
# 
# ## Error Analysis
# For example, notice that our `entity_level_f1` returns `TP`, `FP`, `FN` sets. We can also see that our recall is high, but we have low precision, so let's look at our false positivies, `FP`.

# In[38]:


FP


# We can see that there are actually only a few documents that are causing us problems. In particular, we see that `BC546-D` is giving us many false positives. So, let's inspect one of those candidates. 

# In[39]:


from fonduer.visualizer import *
from hardware_utils import entity_to_candidates
vis = Visualizer(pdf_path)

# Get a list of candidates that match the FN[10] entity
test_cands = session.query(Part_Attr).filter(Part_Attr.split == 2).all()
fp_cands = entity_to_candidates(FP[10], test_cands)
# Display a candidate
vis.display_candidates([fp_cands[0]])


# Here, the candidates are boxed in blue. We see that the `200` falls within the range of numbers that our matcher for storage temperature allows to match. By inspecting candidates like this, or just by examining the problematic PDFs directly, we can notice some patterns that we can exploit as new labeling functions.

# In[40]:


# Get a list of candidates that match the FN[10] entity
test_cands = session.query(Part_Attr).filter(Part_Attr.split == 2).all()
fp_cands = entity_to_candidates(FP[40], test_cands)

# # Display this candidate
vis.display_candidates([fp_cands[0]])


# ## Iteratively Improving Labeling Functions
# 
# From this error analysis, we may notice two important things. First, our original set of labeling functions had no labeling functions that labeled candidates a negative. This resulted in most skewing the models to accept most candidates, and hurt our precision. Second, we have now noticed that we need to focus on negatively labeling numbers that pass through our storage temperature matchers, but are not related to storage temperature.
# 
# Below are a set of negative labeling functions that capture some of these patterns. For example, we label candidates an negative if the number is aligned with attributes that are not related to storage temperature, if a candidate represents a typical value, rather than a maximum value, if a temperature value is found outside of a table, and other intuitive patterns we noticed when carefully inspecting our false positives.

# In[41]:


def LF_test_condition_aligned(c):
    return -1 if overlap(
        ['test', 'condition'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_collector_aligned(c):
    return -1 if overlap(
        ['collector', 'collector-current', 'collector-base', 'collector-emitter'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_current_aligned(c):
    ngrams = get_aligned_ngrams(c.attr)
    return -1 if overlap(
        ['current', 'dc', 'ic'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_voltage_row_temp(c):
    ngrams = get_aligned_ngrams(c.attr)
    return -1 if overlap(
        ['voltage', 'cbo', 'ceo', 'ebo', 'v'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_voltage_row_part(c):
    ngrams = get_aligned_ngrams(c.part)
    return -1 if overlap(
        ['voltage', 'cbo', 'ceo', 'ebo', 'v'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_typ_row(c):
    return -1 if overlap(
        ['typ', 'typ.'],
        list(get_row_ngrams(c.attr))) else 0

def LF_complement_left_row(c):
    return -1 if (
        overlap(['complement','complementary'], 
        chain.from_iterable([get_row_ngrams(c.part), get_left_ngrams(c.part, window=10)]))) else 0


def LF_too_many_numbers_row(c):
    num_numbers = list(get_row_ngrams(c.attr, attrib="ner_tags")).count('number')
    return -1 if num_numbers >= 3 else 0

def LF_temp_on_high_page_num(c):
    return -1 if c.attr.get_attrib_tokens('page')[0] > 2 else 0

def LF_temp_outside_table(c):
    return -1 if not c.attr.sentence.is_tabular() is None else 0

def LF_not_temp_relevant(c):
    return -1 if not overlap(
        ['storage','temperature','tstg','stg', 'ts'],
        list(get_aligned_ngrams(c.attr))) else 0


# Then, we can add these to our list of labeling functions

# In[42]:


stg_temp_lfs_2 = [LF_test_condition_aligned,
                 LF_collector_aligned,
                 LF_current_aligned,
                 LF_voltage_row_temp,
                 LF_voltage_row_part,
                 LF_typ_row,
                 LF_complement_left_row,
                 LF_too_many_numbers_row,
                 LF_temp_on_high_page_num,
                 LF_temp_outside_table,
                 LF_not_temp_relevant
                ]


# And rerun labeling. Importantly, this time we set `clear=False`, `update_keys=True` and `update_values=True` to reflect the fact that we are adding new labeling functions, but do not want to throw away the computations already performed in the previous iteration.

# In[43]:


labeler = BatchLabelAnnotator(Part_Attr, lfs = stg_temp_lfs_2)
L_train = labeler.apply(split=0, clear=False, update_keys=True, update_values=True, parallelism=PARALLEL)
print(L_train.shape)


# Now, we can rerun probablistic relation classification, the same way we did above. We start with the generative model.

# In[44]:


gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
train_marginals = gen_model.marginals(L_train)


# Next, we rerun the discriminitive model and see that our score has improved significantly to about 80 F1 points.

# In[45]:


disc_model = SparseLogisticRegression()
disc_model.train(F_train, train_marginals, n_epochs=200, lr=0.001)


# In[46]:


test_candidates = [F_test.get_candidate(session, i) for i in range(F_test.shape[0])]
test_score = disc_model.predictions(F_test)
true_pred = [test_candidates[_] for _ in np.nditer(np.where(test_score > 0))]
# tp, fp, tn, fn = disc_model.score(session, F_test, L_gold_test)
(TP, FP, FN) = entity_level_f1(true_pred, gold_file, ATTRIBUTE, test_docs, parts_by_doc=parts_by_doc)


# Using these new LFs, we've significantly improved precision and lowered our number of false positives for an F1 score of about 90.
