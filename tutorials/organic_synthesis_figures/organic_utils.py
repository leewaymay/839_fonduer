from __future__ import print_function
from __future__ import division
from builtins import range
import codecs
import csv

from fonduer.lf_helpers import *
from fonduer.snorkel.models import GoldLabel, GoldLabelKey
from fonduer.snorkel.utils import ProgressBar

from fuzzywuzzy import fuzz
FUZZY_SCORE = 70

def get_gold_dict(filename,
                  doc_on=True,
                  org_on=True,
                  fig_on=True,
                  attribute=None,
                  docs=None):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, organic, attr, figure) = row
            if docs is None or doc.upper() in docs:
                if attribute and attr != attribute:
                    continue
                if not figure or not organic:
                    continue
                else:
                    key = []
                    if doc_on: key.append(doc)
                    if org_on: key.append(organic)
                    if fig_on: key.append(figure)
                    gold_dict.add(tuple(key))
    return gold_dict


def load_organic_labels(session,
                         candidate_class,
                         filename,
                         attrib,
                         annotator_name='gold'):

    ak = session.query(GoldLabelKey).filter(
        GoldLabelKey.name == annotator_name).first()
    if ak is None:
        ak = GoldLabelKey(name=annotator_name)
        session.add(ak)
        session.commit()

    candidates = session.query(candidate_class).all()
    gold_dict = get_gold_dict(filename, attribute=attrib)
    cand_total = len(candidates)
    print('Loading', cand_total, 'candidate labels')
    pb = ProgressBar(cand_total)
    labels = []
    for i, c in enumerate(candidates):
        pb.bar(i)
        org = c[0]
        fig = c[1]
        doc_name = org.sentence.document.name
        organic_name = org.text
        figure_src = fig.url
        # context_stable_ids = '~~'.join([i.stable_id for i in c.get_contexts()])
        label = session.query(GoldLabel).filter(GoldLabel.key == ak).filter(
            GoldLabel.candidate == c).first()
        if label is None:
            for t in gold_dict:
                if figure_src == t[2] and fuzz.ratio(t[1], organic_name) >= FUZZY_SCORE:
                    label = GoldLabel(candidate=c, key=ak, value=1)
                    break;

            if label is None:
                label = GoldLabel(candidate=c, key=ak, value=-1)
            session.add(label)
            labels.append(label)
    session.commit()
    pb.close()

    session.commit()
    print("AnnotatorLabels created: %s" % (len(labels), ))


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)

    pred_true = set()
    gold_true = set()
    TP = set()
    for p in pred:
        for g in gold:
            if p[2] == g[2] and fuzz.ratio(p[1], g[1]) >= FUZZY_SCORE:
                pred_true.add(p)
                gold_true.add(g)
                TP.add((p[2],p[1],g[1]))

    FP = pred.difference(pred_true)
    FN = gold.difference(gold_true)
    # TP = pred.intersection(gold)
    # FP = pred.difference(gold)
    # FN = gold.difference(pred)

    return (TP, FP, FN, pred_true, gold_true)


def entity_level_f1(candidates,
                    gold_file,
                    attribute=None,
                    corpus=None,
                    parts_by_doc=None):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = os.environ['FONDUERHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    docs = [(doc.name).upper() for doc in corpus] if corpus else None
    fig_on = (attribute is not None)
    gold_set = get_gold_dict(
        gold_file,
        docs=docs,
        doc_on=True,
        org_on=True,
        fig_on=fig_on,
        attribute=attribute)
    if len(gold_set) == 0:
        print("Gold set is empty.")
        return
    # Turn CandidateSet into set of tuples
    print("Preparing candidates...")
    pb = ProgressBar(len(candidates))
    entities = set()
    for i, c in enumerate(candidates):
        pb.bar(i)
        org = c[0].text
        doc = c[0].sentence.document.name.upper()
        if attribute:
            fig = c[1].url
            entities.add((doc, org, fig))
        else:
            entities.add((doc, org))
        # for p in get_implied_parts(org, doc, parts_by_doc):
        #     if attribute:
        #         entities.add((doc, p, fig))
        #     else:
        #         entities.add((doc, p))
    pb.close()
    print(len(entities))

    (TP_set, FP_set, FN_set, pred_true_set, gold_true_set) = entity_confusion_matrix(entities, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)
    gold_true = len(gold_true_set)
    pred_true = len(pred_true_set)

    prec = pred_true / (pred_true + FP) if TP + FP > 0 else float('nan')
    rec = gold_true / (gold_true + FN) if TP + FN > 0 else float('nan')
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float('nan')
    print("========================================")
    print("Scoring on Entity-Level Gold Data")
    print("========================================")
    print("Total Gold labels      {}".format(len(gold_set)))
    print("Total Predicted labels {}".format(len(entities)))
    print("Corpus Precision {:.3}".format(prec))
    print("Corpus Recall    {:.3}".format(rec))
    print("Corpus F1        {:.3}".format(f1))
    print("----------------------------------------")
    print("TP: {}(gold),{}(pred) | FP: {} | FN: {}".format(gold_true, pred_true, FP, FN))
    print("========================================\n")
    return [sorted(list(x)) for x in [TP_set, FP_set, FN_set]]


def get_implied_parts(part, doc, parts_by_doc):
    yield part
    if parts_by_doc:
        for p in parts_by_doc[doc]:
            if p.startswith(part) and len(part) >= 4:
                yield p


def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple((c[0].sentence.document.name.upper(), c[0].text, c[1].url))
        # c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches

