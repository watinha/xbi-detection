import random, arff

from sklearn import tree, svm, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV,GroupKFold,cross_validate
from functools import reduce

from pipeline import Pipeline
from pipeline.loader.arff_loader import ArffLoader
from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.extractor.crosscheck_extractor import CrossCheckExtractor
from pipeline.extractor.browserninja import *
from pipeline.extractor.browserninja.font_family_extractor import FontFamilyExtractor
from pipeline.extractor.browserninja.relative_position_extractor import RelativePositionExtractor
from pipeline.feature_selection import FeatureSelection
from pipeline.classifier.classifier_tunning import ClassifierTunning
from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

features = [
    #'URL', 'id', 'tagName',
    'childsNumber', 'textLength',
    #'basePlatform', 'targetPlatform', 'baseBrowser', 'targetBrowser',
    'baseDPI', 'targetDPI',
    #'baseScreenshot', 'targetScreenshot',
    'baseX', 'targetX', 'baseY', 'targetY',
    'baseHeight', 'targetHeight', 'baseWidth', 'targetWidth',
    'baseParentX', 'targetParentX', 'baseParentY', 'targetParentY',
    'imageDiff', 'chiSquared',
    'baseDeviceWidth', 'targetDeviceWidth', 'baseViewportWidth', 'targetViewportWidth',
    #'xpath', 'baseXpath', 'targetXpath',
    'phash',
    'basePreviousSiblingLeft', 'targetPreviousSiblingLeft',
    'basePreviousSiblingTop', 'targetPreviousSiblingTop',
    'baseNextSiblingLeft', 'targetNextSiblingLeft',
    'baseNextSiblingTop', 'targetNextSiblingTop',
    'baseTextNodes', 'targetTextNodes',
    #'baseFontFamily', 'targetFontFamily'
]

random.seed(42)
pipeline = Pipeline([
    ArffLoader(),
    #XBIExtractor(features, 'Result'),
    #CrossCheckExtractor('Result'),
    BrowserNinjaCompositeExtractor('Result',
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
            #FontFamilyExtractor(),
            RelativePositionExtractor(),
            PlatformExtractor()
        ]),
    #FeatureSelection(SelectKBest(f_classif, k=20)),
    ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
            'n_estimators': [2, 5, 10, 15],
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 30, 50, 100, None],
            'min_samples_split': [2, 3, 10, 30],
            'min_samples_leaf': [1, 3, 5],
            'max_features': [5, 10, 'auto'],
            'class_weight': [None, 'balanced']
        }, cv=GroupKFold(n_splits=3)),
        ensemble.RandomForestClassifier(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(svm.LinearSVC(), {
    #        #'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #        #'kernel': ['linear'],
    #        'C': [1, 10, 100],
    #        #'degree': [1, 2, 3],
    #        #'coef0': [0, 10, 100],
    #        'tol': [0.001, 0.1, 1],
    #        'class_weight': ['balanced', None],
    #        'max_iter': [2000]
    #    }, cv=GroupKFold(n_splits=3)),
    #    svm.LinearSVC(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
    #        'criterion': ["gini", "entropy"],
    #        'max_depth': [3, 5, 20, None],
    #        'min_samples_split': [2, 7, 13],
    #        'class_weight': [None, 'balanced'],
    #        #'max_features': [5, None],
    #        'max_features': [5, 10, None],
    #        'min_samples_leaf': [1, 5, 10]
    #    #}, cv=5),
    #    }, cv=GroupKFold(n_splits=3)),
    #    tree.DecisionTreeClassifier(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(MLPClassifier(), {
    #        'hidden_layer_sizes': [5, 10, 30],
    #        'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #        'solver': ['lbfgs', 'sgd', 'adam'],
    #        'alpha': [0.0001, 0.01, 0.1],
    #        'max_iter': [2000],
    #        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #        'random_state': [42]
    #    }, cv=GroupKFold(n_splits=3)),
    #    MLPClassifier(random_state=42), 'URL'),
    GroupKFoldCV(GroupKFold(n_splits=10), 'URL', cross_validate)
])
result = pipeline.execute(open('data/dataset-filtered-040919-noblogspot.arff').read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('X dimensions:' + str(result['X'].shape))
print('Trainning F1: ' + str(result['score']['train_f1_macro']))
print('Test      F1: ' + str(result['score']['test_f1_macro']))
print('Trainning F1: %f' % (reduce(lambda x,y: x+y, result['score']['train_f1_macro']) / 10))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1_macro']) / 10))
#print('Trainning Precision: ' + str(result['score']['train_precision_macro']))
#print('Test      Precision: ' + str(result['score']['test_precision_macro']))
#print('Trainning Recall: ' + str(result['score']['train_recall_macro']))
#print('Test      Recall: ' + str(result['score']['test_recall_macro']))

#print('--- Error analysis ---')
#model = result['model']
#dataset = result['data']
#X = result['X']
#y = result['y']
#model.fit(X, y)
#r = model.predict(X)
#diff = [ dataset[i].tolist() for i in range(0,len(r)) if r[i] != y[i] ]
#diff_x = [ X[i].tolist() for i in range(0,len(r)) if r[i] != y[i] ]
#arff_data = {
#    'attributes': result['attributes'],
#    'description': result['description'],
#    'relation': result['relation'],
#    'data': diff
#}
#print(arff.dumps(arff_data))
#print(diff_x)
