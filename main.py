import random, arff, sys

import pandas as pd

from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline as Pipe
from functools import reduce

from config import get_extractor, get_classifier, get_sampler
from pipeline import Pipeline
from pipeline.loader.arff_loader import ArffLoader
from pipeline.feature_selection import FeatureSelection
from pipeline.preprocessing import Preprocessor
from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

assert len(sys.argv) == 5, 'The script accepts 4 parameters: feature extractor (browserbite|crosscheck|browserninja1|browserninja2), classifier (randomforest|svm|dt|nn), type of xbi (internal|external) and K value'

random.seed(42)

class_attr = sys.argv[3]
k = int(sys.argv[4])
extractor_name = sys.argv[1]
classifier_name = sys.argv[2]
rankings = []

(extractor, features, nfeatures, max_features) = get_extractor(
        extractor_name, class_attr)
classifier = get_classifier(classifier_name)
sampler = get_sampler()

class GroupFolds:

    def __init__ (self, cv, groups):
        self._cv = cv
        self._groups = groups

    def split (self, X, y):
        for train_index, test_index in self._cv.split(X, y, self._groups):
            yield (train_index, test_index)


def cross_val_score_using_sampling(model, X, y, cv, groups, scoring):
    fscore = []
    precision = []
    recall = []
    roc = []
    best_fscore = []
    best_precision = []
    best_recall = []
    best_roc = []
    model2 = None
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = [i for i in groups if groups.index(i) in train_index]

        X_samp, y_samp = sampler.fit_resample(X_train, y_train)
        groups_samp = [ groups_train[X_train.tolist().index(row)] for row in X_samp.tolist() ]

        print('Model trainning with: X (%s)' % (str(X_samp.shape)))
        model.fit(X_samp, y_samp, groups=groups_samp)

        selector = model.best_estimator_.named_steps['selector']
        rankings.append(selector.get_support(indices=False))

        y_pred = model.predict(X_test)

        fscore.append(metrics.f1_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        roc.append(metrics.roc_auc_score(y_test, y_pred))

        if classifier_name == 'svm':
            cv = GroupFolds(GroupShuffleSplit(n_splits=10), groups_samp)
            cclassifier = CalibratedClassifierCV(model.best_estimator_.named_steps['classifier'], cv=cv)
            model2 = Pipe([('selector', model.best_estimator_.named_steps['selector']), ('classifier', cclassifier)])
            model2.fit(X_samp, y_samp)
        else:
            model2 = model

        y_pred = model2.predict_proba(X_samp)
        probability = y_pred[:,1]

        best_roc.append(metrics.roc_auc_score(y_samp, probability))
        precision2, recall2, threasholds = metrics.precision_recall_curve(y_samp, probability)
        best_f = 0
        best_r = 0
        best_p = 0
        threashold = 0
        for i in range(len(precision2)):
            new_fscore = 2 * precision2[i] * recall2[i] / (precision2[i] + recall2[i])
            if new_fscore > best_f:
                best_f = new_fscore
                threshold = threasholds[i]

        y_pred = model2.predict_proba(X_test)
        y_pred = [ 0 if y < threshold else 1 for y in y_pred[:,1]]
        best_f = metrics.f1_score(y_test, y_pred)
        best_p = metrics.precision_score(y_test, y_pred)
        best_r = metrics.recall_score(y_test, y_pred)

        best_fscore.append(best_f)
        best_precision.append(best_p)
        best_recall.append(best_r)

    return { 'test_f1': fscore, 'test_precision': precision, 'test_recall': recall, 'test_roc_auc': roc,
            'best_f1': best_fscore, 'best_precision': best_precision, 'best_recall': best_recall, 'best_roc': best_roc }

groupcv = None
groupcv = GroupKFoldCV(GroupShuffleSplit(n_splits=10, random_state=42), 'URL', cross_val_score_using_sampling) # 72 websites + 3 platform comparison

preprocessor = Preprocessor()
approach = '%s-%s-%s-k%s' % (extractor_name, classifier_name, class_attr, str(k))
print('running --- %s...' % (approach))
pipeline = Pipeline([
    ArffLoader(), XBIExtractor(features, class_attr),
    extractor, preprocessor, classifier, groupcv
])
result = pipeline.execute(open('data/19112021/dataset.classified.hist.img.%s.arff' % (class_attr)).read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('K: ' + str(k))
print('X dimensions:' + str(result['X'].shape))
print('Test     ROC: %f' % (reduce(lambda x,y: x+y, result['score']['test_roc_auc']) / 10))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1']) / 10))
print('Test      F1: ' + str(result['score']['test_f1']))
print('Test      Precision: ' + str(result['score']['test_precision']))
print('Test      Recall: ' + str(result['score']['test_recall']))
print('Best      F1: ' + str(result['score']['best_f1']))
print('Best      F1: %f' % (reduce(lambda x,y: x+y, result['score']['best_f1']) / 10))
print('Best      Precision: ' + str(result['score']['best_precision']))
print('Best      Recall: ' + str(result['score']['best_recall']))
print('Best     ROC: ' + str(result['score']['best_roc']))
print('Best     ROC: %f' % (reduce(lambda x, y: x+y, result['score']['best_roc']) / 10))

fscore = result['score']['best_f1']
precision = result['score']['best_precision']
recall = result['score']['best_recall']
roc = result['score']['test_roc_auc']

try:
    fscore_csv = pd.read_csv('results/fscore-%s.csv' % (class_attr), index_col=0)
    precision_csv = pd.read_csv('results/precision-%s.csv' % (class_attr), index_col=0)
    recall_csv = pd.read_csv('results/recall-%s.csv' % (class_attr), index_col=0)
    roc_csv = pd.read_csv('results/roc-%s.csv' % (class_attr), index_col=0)
except:
    fscore_csv = pd.DataFrame()
    precision_csv = pd.DataFrame()
    recall_csv = pd.DataFrame()
    roc_csv = pd.DataFrame()

fscore_csv.loc[:, approach] = fscore
precision_csv.loc[:, approach] = precision
recall_csv.loc[:, approach] = recall
roc_csv.loc[:, approach] = roc

fscore_csv.to_csv('results/fscore-%s.csv' % (class_attr))
precision_csv.to_csv('results/precision-%s.csv' % (class_attr))
recall_csv.to_csv('results/recall-%s.csv' % (class_attr))
roc_csv.to_csv('results/roc-%s.csv' % (class_attr))

try:
    features_csv = pd.read_csv('results/features-%s.csv' % (class_attr), index_col=0)
except:
    features_csv = pd.DataFrame(columns=result['features'])

features_len = features_csv.shape[1]
print(rankings)
if extractor_name == 'browserninja2':
    for i in range(len(rankings)):
        features_csv.loc['%s-k%d-%d' % (classifier_name, k, (i + features_len)), :] = rankings[i]
    features_csv.to_csv('results/features-%s.csv' % (class_attr))
