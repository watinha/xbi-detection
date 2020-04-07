import random, arff, sys

from sklearn import tree, svm, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV,GroupKFold,cross_validate
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
from pipeline.classifier.classifier_tunning import ClassifierTunning
from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

assert len(sys.argv) == 4, 'The script accepts 3 parameters: feature extractor (browserbite|crosscheck|browserninja), classifier (randomforest|svm|dt|nn) and type of xbi (internal|external)'

class_attr = sys.argv[3]

extractor = None
if sys.argv[1] == 'browserbite':
    extractor = BrowserbiteExtractor(class_attr)
else if sys.argv[1] == 'crosscheck':
    extractor = CrossCheckExtractor(class_attr)
else:
    extractor = BrowserNinjaCompositeExtractor('internal',
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
            #FontFamilyExtractor(),
            RelativePositionExtractor(),
            PlatformExtractor()
        ])

classifier = None
if sys.argv[1] == 'randomforest':
    classifier = ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
            'n_estimators': [2, 5, 10, 15],
            'criterion': ["gini", "entropy"],
            'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
            'min_samples_split': [1, 10, 30], #'min_samples_split': [2, 3, 10, 30],
            'min_samples_leaf': [1, 5, 10],
            'max_features': [5, 10, 'auto'],
            'class_weight': [None, 'balanced']
        }, cv=GroupKFold(n_splits=3)),
        ensemble.RandomForestClassifier(random_state=42), 'URL'),
else if sys.argv[1] == 'svm':
    classifier = ClassifierTunning(GridSearchCV(svm.LinearSVC(), {
            #'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            #'kernel': ['linear'],
            'C': [1, 10, 100],
            #'degree': [1, 2, 3],
            #'coef0': [0, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None],
            'max_iter': [2000]
        }, cv=GroupKFold(n_splits=3)),
        svm.LinearSVC(random_state=42), 'URL'),
else if sys.argv[1] == 'dt':
    classifier = ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
            'criterion': ["gini", "entropy"],
            'max_depth': [5, 10, None],
            'min_samples_split': [10, 30, 50],
            'class_weight': [None, 'balanced'],
            #'max_features': [5, None],
            #'max_features': [5, 10, None],
            'min_samples_leaf': [1, 5, 10]
        #}, cv=5),
        }, cv=GroupKFold(n_splits=3)),
        tree.DecisionTreeClassifier(random_state=42), 'URL'),
else:
    classifier = ClassifierTunning(GridSearchCV(MLPClassifier(), {
            'hidden_layer_sizes': [5, 10, 30],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.01, 0.1],
            'max_iter': [2000],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'random_state': [42]
        }, cv=GroupKFold(n_splits=3)),
        MLPClassifier(random_state=42), 'URL'),

pipeline = Pipeline([
    ArffLoader(), extractor, classifier,
    GroupKFoldCV(GroupKFold(n_splits=10), 'URL', cross_validate)])

result = pipeline.execute(open('data/07042020/07042020-dataset.binary.hist.arff').read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('X dimensions:' + str(result['X'].shape))
print('Trainning F1: ' + str(result['score']['train_f1_macro']))
print('Test      F1: ' + str(result['score']['test_f1_macro']))
print('Trainning F1: %f' % (reduce(lambda x,y: x+y, result['score']['train_f1_macro']) / 10))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1_macro']) / 10))
print('Test      Precision: ' + str(result['score']['test_precision_macro']))
print('Test      Recall: ' + str(result['score']['test_recall_macro']))
