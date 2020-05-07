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
from pipeline.preprocessing import Preprocessor
from pipeline.classifier.classifier_tunning import ClassifierTunning
from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

assert len(sys.argv) == 5, 'The script accepts 4 parameters: feature extractor (browserbite|crosscheck|browserninja1|browserninja2), classifier (randomforest|svm|dt|nn), type of xbi (internal|external) and K value'

class_attr = sys.argv[3]
k = int(sys.argv[4])

extractor = None
if sys.argv[1] == 'browserbite':
    extractor = BrowserbiteExtractor(class_attr)
elif sys.argv[1] == 'crosscheck':
    extractor = CrossCheckExtractor(class_attr)
elif sys.argv[1] == 'browserninja1':
    extractor = BrowserNinjaCompositeExtractor(class_attr,
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
        ])
elif sys.argv[1] == 'browserninja2':
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
if sys.argv[2] == 'randomforest':
    if sys.argv[1] == 'crosscheck':
        classifier = ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
                'n_estimators': [5, 10, 15],
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
                'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
                'min_samples_leaf': [1, 5, 10],
                'class_weight': [None, 'balanced']
            }, cv=GroupKFold(n_splits=3)),
            ensemble.RandomForestClassifier(random_state=42), 'URL')
    else:
        classifier = ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
                'n_estimators': [5, 10, 15],
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
                'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
                'min_samples_leaf': [1, 5, 10],
                'max_features': [5, 10, 'auto'],
                'class_weight': [None, 'balanced']
            }, cv=GroupKFold(n_splits=3)),
            ensemble.RandomForestClassifier(random_state=42), 'URL')
elif sys.argv[2] == 'svm':
    classifier = ClassifierTunning(GridSearchCV(svm.SVC(), {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            #'kernel': ['linear'],
            'C': [1, 10, 100],
            'degree': [1, 2, 3],
            'coef0': [0, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None],
            'max_iter': [1000]
        }, cv=GroupKFold(n_splits=3)),
        svm.SVC(random_state=42), 'URL')
elif sys.argv[2] == 'dt':
    if sys.argv[1] == 'crosscheck':
        classifier = ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None],
                'min_samples_split': [10, 30, 50],
                'class_weight': [None, 'balanced'],
                #'max_features': [5, 10, None],
                'min_samples_leaf': [1, 5, 10]
            }, cv=GroupKFold(n_splits=3)),
            tree.DecisionTreeClassifier(random_state=42), 'URL')
    else:
        classifier = ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
                'criterion': ["gini", "entropy"],
                'max_depth': [5, 10, None],
                'min_samples_split': [10, 30, 50],
                'class_weight': [None, 'balanced'],
                'max_features': [5, 10, None],
                'min_samples_leaf': [1, 5, 10]
            }, cv=GroupKFold(n_splits=3)),
            tree.DecisionTreeClassifier(random_state=42), 'URL')
else:
    classifier = ClassifierTunning(GridSearchCV(MLPClassifier(), {
            'hidden_layer_sizes': [5, 10, 30],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.01, 0.1],
            'max_iter': [1000],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'random_state': [42]
        }, cv=GroupKFold(n_splits=3)),
        MLPClassifier(random_state=42), 'URL')

preprocessor = Preprocessor()
selector = FeatureSelection(SelectKBest(f_classif, k=k), k=k)
pipeline = Pipeline([
    ArffLoader(), extractor, preprocessor, selector, classifier,
    GroupKFoldCV(GroupKFold(n_splits=10), 'URL', cross_validate)])
result = pipeline.execute(open('data/07042020/07042020-dataset.binary.hist.arff').read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('K: ' + str(k))
print('X dimensions:' + str(result['X'].shape))
print('Test      F1: ' + str(result['score']['test_f1']))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1']) / 10))
print('Test      Precision: ' + str(result['score']['test_precision']))
print('Test      Recall: ' + str(result['score']['test_recall']))
if k == 300 and (sys.argv[2] == 'dt' or sys.argv[2] == 'randomforest'):
    result['model'].fit(result['X'], result['y'])
    for i in range(len(result['features'])):
        print('%s: %f' % (result['features'][i], result['model'].feature_importances_[i]))
