from sklearn import tree
from sklearn.model_selection import GridSearchCV,GroupKFold,cross_validate

from pipeline import Pipeline
from pipeline.loader.arff_loader import ArffLoader
from pipeline.extractor.xbi_extractor import XBIExtractor
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

pipeline = Pipeline([
    ArffLoader(),
    XBIExtractor(features, 'Result'),
    ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
        'criterion': ["gini", "entropy"],
        'max_depth': [10, 60, 100, None],
        'min_samples_split': [2, 30, 100],
        'class_weight': [None, 'balanced']
    }, cv=2),
    tree.DecisionTreeClassifier(random_state=42)),
    GroupKFoldCV(GroupKFold(n_splits=10), 'URL', cross_validate)
])
result = pipeline.execute(open('data/dataset-040919.arff').read())
print('Trainning F1: ' + str(result['score']['train_f1_macro']))
print('Test      F1: ' + str(result['score']['test_f1_macro']))
print('Trainning F1: %f' % reduce(lambda x,y: x+y, result['score']['train_f1_macro']) / 10)
print('Test      F1: %f' % reduce(lambda x,y: x+y, result['score']['test_f1_macro']) / 10)
#print('Trainning Precision: ' + str(result['score']['train_precision_macro']))
#print('Test      Precision: ' + str(result['score']['test_precision_macro']))
#print('Trainning Recall: ' + str(result['score']['train_recall_macro']))
#print('Test      Recall: ' + str(result['score']['test_recall_macro']))
