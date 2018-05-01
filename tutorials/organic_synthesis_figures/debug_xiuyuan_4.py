import os
import sys
# os.system('bash ./set_env.sh')
PARALLEL = 1 # assuming a quad-core machine
ATTRIBUTE = "organic_figure"

os.environ['FONDUERHOME'] = '/home/xiuyuan/private/839/fonduer_new/839_fonduer/'
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://postgres:112233@localhost:5432/' + os.environ['FONDUERDBNAME']


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import candidate_subclass

Org_Fig = candidate_subclass('Org_Fig', ['product','figure'])

from fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + 'tutorials/organic_synthesis_figures/data/html/'
pdf_path = os.environ['FONDUERHOME'] + 'tutorials/organic_synthesis_figures/data/pdf/'



# load gold label
from tutorials.organic_synthesis_figures.organic_utils import load_organic_labels

gold_file = os.environ['FONDUERHOME'] + 'tutorials/organic_synthesis_figures/data/organic_gold.csv'
load_organic_labels(session, Org_Fig, gold_file, ATTRIBUTE ,annotator_name='gold')

from tutorials.organic_synthesis_figures import organic_lfs


from fonduer import BatchLabelAnnotator
#
labeler = BatchLabelAnnotator(Org_Fig, lfs = organic_lfs.org_fig_lfs)
L_train = labeler.apply(split=0, clear=True, parallelism=PARALLEL)
# print(L_train.shape)
#
# L_train.get_candidate(session, 0)

# # Applying the Labeling Functions
from fonduer import load_gold_labels
L_gold_train = load_gold_labels(session, annotator_name='gold', split=0)
L_train.lf_stats(L_gold_train)
