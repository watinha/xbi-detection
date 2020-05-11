import random, arff, sys

import pandas as pd

from imblearn.under_sampling import TomekLinks, ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn import tree, svm, ensemble
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV,GroupKFold,GroupShuffleSplit,cross_validate
from functools import reduce

from pipeline import Pipeline
from pipeline.loader.arff_loader import ArffLoader
from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.extractor.crosscheck_extractor import CrossCheckExtractor
from pipeline.extractor.browserbite_extractor import BrowserbiteExtractor
from pipeline.extractor.browserninja import *
from pipeline.extractor.browserninja.font_family_extractor import FontFamilyExtractor
from pipeline.extractor.browserninja.relative_position_extractor import RelativePositionExtractor
from pipeline.feature_selection import FeatureSelection
from pipeline.preprocessing import Preprocessor
from pipeline.classifier.classifier_tunning import ClassifierTunning
from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

assert len(sys.argv) == 5, 'The script accepts 4 parameters: feature extractor (browserbite|crosscheck|browserninja1|browserninja2), classifier (randomforest|svm|dt|nn), type of xbi (internal|external) and K value'

class_attr = sys.argv[3]
k = int(sys.argv[4])
extractor_name = sys.argv[1]
classifier_name = sys.argv[2]

extractor = None
if extractor_name == 'browserbite':
    extractor = BrowserbiteExtractor(class_attr)
elif extractor_name == 'crosscheck':
    extractor = CrossCheckExtractor(class_attr)
elif extractor_name == 'browserninja1':
    extractor = BrowserNinjaCompositeExtractor(class_attr,
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
        ])
elif extractor_name == 'browserninja2':
    extractor = BrowserNinjaCompositeExtractor(class_attr,
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
            RelativePositionExtractor(),
            PlatformExtractor()
        ])
else:
    extractor = BrowserNinjaCompositeExtractor(class_attr,
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
            FontFamilyExtractor(),
            RelativePositionExtractor(),
            PlatformExtractor()
        ])

classifier = None
if classifier_name == 'randomforest':
    if extractor_name == 'crosscheck':
        classifier = ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
                'n_estimators': [5, 10, 15],
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
                'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
                'min_samples_leaf': [1, 5, 10],
                'class_weight': [None, 'balanced']
            }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
            ensemble.RandomForestClassifier(random_state=42), 'URL')
    else:
        classifier = ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
                'n_estimators': [5, 10, 15],
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
                'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
                'min_samples_leaf': [1, 5, 10],
                'max_features': [3, 5, 10, 'auto'],
                'class_weight': [None, 'balanced']
            }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
            ensemble.RandomForestClassifier(random_state=42), 'URL')
elif classifier_name == 'svm':
    classifier = ClassifierTunning(GridSearchCV(svm.SVC(), {
    #classifier = ClassifierTunning(GridSearchCV(svm.LinearSVC(), {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': [1, 2, 3],
            'coef0': [0, 10, 100],
        #    'dual': [False],
            'C': [1, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None],
            'max_iter': [5000]
        }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
        svm.SVC(random_state=42), 'URL')
        #svm.LinearSVC(random_state=42), 'URL')
elif classifier_name == 'dt':
    if extractor_name == 'crosscheck':
        classifier = ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None],
                'min_samples_split': [10, 30, 50],
                'class_weight': [None, 'balanced'],
                #'max_features': [5, 10, None],
                'min_samples_leaf': [1, 5, 10]
            }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
            tree.DecisionTreeClassifier(random_state=42), 'URL')
    else:
        classifier = ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None],
                'min_samples_split': [10, 30, 50],
                'class_weight': [None, 'balanced'],
                'max_features': [3, 5, 10, None],
                'min_samples_leaf': [1, 5, 10]
            }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
            tree.DecisionTreeClassifier(random_state=42), 'URL')
else:
    classifier = ClassifierTunning(GridSearchCV(MLPClassifier(), {
            'hidden_layer_sizes': [5, 10, 30],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
#            'solver': ['lbfgs', 'sgd', 'adam'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.01, 0.1],
            'max_iter': [1000],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'random_state': [42]
        }, cv=GroupShuffleSplit(n_splits=3, random_state=42)),
        MLPClassifier(random_state=42), 'URL')

sampler = TomekLinks()

def cross_val_score_using_sampling(model, X, y, cv, groups, scoring):
    fscore = []
    precision = []
    recall = []
    roc = []
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_samp, y_samp = sampler.fit_sample(X_train, y_train)
        model.fit(X_samp, y_samp)
        y_pred = model.predict(X_test)
        fscore.append(metrics.f1_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        roc.append(metrics.roc_auc_score(y_test, y_pred))
    return { 'test_f1': fscore, 'test_precision': precision, 'test_recall': recall, 'test_roc_auc': roc }

groupcv = None
if (classifier_name == 'svm' or classifier_name == 'nn'):
    groupcv = GroupKFoldCV(GroupShuffleSplit(n_splits=10, random_state=42), 'URL', cross_val_score_using_sampling)
else:
    groupcv = GroupKFoldCV(GroupShuffleSplit(n_splits=10, random_state=42), 'URL', cross_validate)

preprocessor = Preprocessor()
selector = FeatureSelection(SelectKBest(f_classif, k=k), k=k)
approach = '%s-%s-%s-k%s' % (extractor_name, classifier_name, class_attr, str(k))

print('running --- %s...' % (approach))
pipeline = Pipeline([
    ArffLoader(), extractor, preprocessor, classifier, groupcv])
result = pipeline.execute(open('data/07042020/07042020-dataset.binary.hist.arff').read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('K: ' + str(k))
print('X dimensions:' + str(result['X'].shape))
print('Test     ROC: %f' % (reduce(lambda x,y: x+y, result['score']['test_roc_auc']) / 10))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1']) / 10))
print('Test      F1: ' + str(result['score']['test_f1']))
print('Test      Precision: ' + str(result['score']['test_precision']))
print('Test      Recall: ' + str(result['score']['test_recall']))
if k == 300 and (classifier_name == 'dt' or classifier_name == 'randomforest'):
    result['model'].fit(result['X'], result['y'])
    for i in range(len(result['features'])):
        print('%s: %f' % (result['features'][i], result['model'].feature_importances_[i]))

fscore = result['score']['test_f1']
precision = result['score']['test_precision']
recall = result['score']['test_recall']
roc = result['score']['test_roc_auc']
fscore_csv = pd.read_csv('results/fscore-%s.csv' % (class_attr), index_col=0)
precision_csv = pd.read_csv('results/precision-%s.csv' % (class_attr), index_col=0)
recall_csv = pd.read_csv('results/recall-%s.csv' % (class_attr), index_col=0)
roc_csv = pd.read_csv('results/roc-%s.csv' % (class_attr), index_col=0)

fscore_csv[approach] = fscore
precision_csv[approach] = precision
recall_csv[approach] = recall
roc_csv[approach] = roc

fscore_csv.to_csv('results/fscore-%s.csv' % (class_attr))
precision_csv.to_csv('results/precision-%s.csv' % (class_attr))
recall_csv.to_csv('results/recall-%s.csv' % (class_attr))
roc_csv.to_csv('results/roc-%s.csv' % (class_attr))
