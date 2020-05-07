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

features = [
    #'URL', 'id', 'tagName',
    'childsNumber', 'textLength',
    #'basePlatform', 'targetPlatform', 'baseBrowser', 'targetBrowser',
    #'baseDPI', 'targetDPI',
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
    #'baseFontFamily', 'targetFontFamily',
    'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
    'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
    'target_bin1', 'target_bin2', 'target_bin3', 'target_bin4', 'target_bin5',
    'target_bin6', 'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10',
    'diff_bin01', 'diff_bin02', 'diff_bin03', 'diff_bin04', 'diff_bin05',
    'diff_bin11', 'diff_bin12', 'diff_bin13', 'diff_bin14', 'diff_bin15',
    'diff_bin21', 'diff_bin22', 'diff_bin23', 'diff_bin24', 'diff_bin25',
    'diff_bin31', 'diff_bin32', 'diff_bin33', 'diff_bin34', 'diff_bin35',
    'diff_bin41', 'diff_bin42', 'diff_bin43', 'diff_bin44', 'diff_bin45'
]

random.seed(42)
pipeline = Pipeline([
    ArffLoader(),
    #XBIExtractor(features, 'internal'),
    #CrossCheckExtractor('internal'),
    #BrowserbiteExtractor('internal'),
    BrowserNinjaCompositeExtractor('internal',
        extractors=[
            ComplexityExtractor(),
            ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
            FontFamilyExtractor(),
            RelativePositionExtractor(),
            PlatformExtractor()
        ]),
    Preprocessor(),
    FeatureSelection(SelectKBest(f_classif, k=8), k=8),
    #ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
    #        'n_estimators': [2, 5, 10, 15],
    #        'criterion': ["gini", "entropy"],
    #        'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
    #        'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
    #        'min_samples_leaf': [1, 5, 10],
    #        'max_features': [3, 4, 5, 10, 'auto'],
    #        'class_weight': [None, 'balanced']
    #    }, cv=GroupKFold(n_splits=3)),
    #    ensemble.RandomForestClassifier(random_state=42), 'URL'),
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
    ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
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
result = pipeline.execute(open('data/07042020/07042020-dataset.binary.hist.arff').read())
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
