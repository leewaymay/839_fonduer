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





from fonduer import BatchLabelAnnotator
from tutorials.organic_synthesis_figures import organic_lfs

labeler = BatchLabelAnnotator(Org_Fig, lfs = organic_lfs.org_fig_lfs)
L_train = labeler.load_matrix(split=0)
print(L_train.shape)

L_train.get_candidate(session, 0)

# Applying the Labeling Functions
from fonduer import load_gold_labels
L_gold_train = load_gold_labels(session, annotator_name='gold', split=0)
print(L_train.lf_stats(L_gold_train))

L_gold_test = load_gold_labels(session, annotator_name='gold', split=1)


'''

    Training

'''

from fonduer import GenerativeModel

gen_model = GenerativeModel()
gen_model.train(L_train, epochs=500, decay=0.9, step_size=0.001/L_train.shape[0], reg_param=0)
train_marginals = gen_model.marginals(L_train)
print(gen_model.weights.lf_accuracy)
L_test = labeler.apply_existing(split=1)
print("L_DEV:")
print(L_test.shape)


from fonduer import load_gold_labels
L_gold_test = load_gold_labels(session, annotator_name='gold', split=1)
prec, rec, f1 = gen_model.score(L_test, L_gold_test)

print("precision: " + str(prec))
print("recall: " + str(rec))
print("f1: " + str(f1))



print(L_test.lf_stats(L_gold_test, gen_model.weights.lf_accuracy))

