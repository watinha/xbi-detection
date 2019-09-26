import random, arff

from sklearn import tree, svm, ensemble
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
    #'childsNumber', 'textLength',
    #'basePlatform', 'targetPlatform', 'baseBrowser', 'targetBrowser',
    #'baseDPI', 'targetDPI',
    #'baseScreenshot', 'targetScreenshot',
    #'baseX', 'targetX', 'baseY', 'targetY',
    #'baseHeight', 'targetHeight', 'baseWidth', 'targetWidth',
    #'baseParentX', 'targetParentX', 'baseParentY', 'targetParentY',
    #'imageDiff', 'chiSquared',
    #'baseDeviceWidth', 'targetDeviceWidth', 'baseViewportWidth', 'targetViewportWidth',
    #'xpath', 'baseXpath', 'targetXpath',
    #'phash',
    #'basePreviousSiblingLeft', 'targetPreviousSiblingLeft',
    #'basePreviousSiblingTop', 'targetPreviousSiblingTop',
    #'baseNextSiblingLeft', 'targetNextSiblingLeft',
    #'baseNextSiblingTop', 'targetNextSiblingTop',
    #'baseTextNodes', 'targetTextNodes',
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
    #        ImageComparisonExtractor(),
            SizeViewportExtractor(),
            VisibilityExtractor(),
            PositionViewportExtractor(),
    #        FontFamilyExtractor(),
            RelativePositionExtractor()
        ]),
    #FeatureSelection(SelectKBest(f_classif, k=20)),
    #ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
    #    'n_estimators': [5, 10, 100],
    #    'criterion': ["gini", "entropy"],
    #    'max_depth': [10, 50, 100, None],
    #    'min_samples_split': [2, 10, 100],
    #    'class_weight': [None, 'balanced']
    #}, cv=2),
    #ensemble.RandomForestClassifier(random_state=42)),
    #ClassifierTunning(GridSearchCV(svm.LinearSVC(), {
    #    #'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #    #'kernel': ['linear'],
    #    'C': [1, 10, 100],
    #    #'degree': [1, 2, 3],
    #    #'coef0': [0, 10, 100],
    #    'tol': [0.001, 0.1, 1],
    #    'class_weight': ['balanced', None],
    #    'max_iter': [30000]
    #}, cv=2),
    #svm.LinearSVC(random_state=42)),
    ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
        'criterion': ["gini", "entropy"],
        'max_depth': [10, 60, 100, None],
        'min_samples_split': [2, 30, 100],
        'class_weight': [None, 'balanced']
    }, cv=10),
    tree.DecisionTreeClassifier(random_state=42)),
    GroupKFoldCV(GroupKFold(n_splits=10), 'URL', cross_validate)
])
result = pipeline.execute(open('data/classified/external-dataset.arff').read())
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
