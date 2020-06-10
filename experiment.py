import random, arff, sys

from imblearn.under_sampling import TomekLinks, ClusterCentroids
from sklearn import tree, svm, ensemble, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV
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
    #'baseFontFamily', 'targetFontFamily',
    #'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
    #'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
    #'target_bin1', 'target_bin2', 'target_bin3', 'target_bin4', 'target_bin5',
    #'target_bin6', 'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10',
    'diff_bin01', 'diff_bin02', 'diff_bin03', 'diff_bin04', 'diff_bin05',
    'diff_bin11', 'diff_bin12', 'diff_bin13', 'diff_bin14', 'diff_bin15',
    'diff_bin21', 'diff_bin22', 'diff_bin23', 'diff_bin24', 'diff_bin25',
    'diff_bin31', 'diff_bin32', 'diff_bin33', 'diff_bin34', 'diff_bin35',
    'diff_bin41', 'diff_bin42', 'diff_bin43', 'diff_bin44', 'diff_bin45'
]

sampler = TomekLinks()

def cross_val_score_using_sampling(model, X, y, cv, groups, scoring):
    fscore = []
    precision = []
    recall = []
    roc = []
    best_roc = []
    best_fscore = []
    best_recall = []
    best_precision = []
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

        y_pred = model.predict_proba(X_test)
        probability = y_pred[:,1]

        best_roc.append(metrics.roc_auc_score(y_test, probability))
        precision2, recall2, threasholds = metrics.precision_recall_curve(y_test, probability)
        best_f = 0
        best_r = 0
        best_p = 0
        for i in range(len(precision2)):
            new_fscore = 2 * precision2[i] * recall2[i] / (precision2[i] + recall2[i])
            if new_fscore > best_f:
                best_f = new_fscore
                best_r = recall2[i]
                best_p = precision2[i]

        best_fscore.append(best_f)
        best_precision.append(best_p)
        best_recall.append(best_r)

    return { 'test_f1': fscore, 'test_precision': precision, 'test_recall': recall, 'test_roc_auc': roc,
            'best_f1': best_fscore, 'best_precision': best_precision, 'best_recall': best_recall, 'best_roc': best_roc }

rfecv = RFECV(ensemble.RandomForestClassifier(), scoring='f1_macro')
random.seed(42)
pipeline = Pipeline([
    ArffLoader(),
    XBIExtractor(features, 'internal'),
    CrossCheckExtractor('internal'),
    BrowserbiteExtractor('internal'),
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
    FeatureSelection(rfecv),
    #FeatureSelection(SelectKBest(f_classif, k=8), k=8),
    ClassifierTunning(GridSearchCV(ensemble.RandomForestClassifier(), {
            'n_estimators': [2, 5, 10, 15],
            'criterion': ["gini", "entropy"],
            'max_depth': [5, 10, None], #'max_depth': [5, 10, 30, 50, None],
            'min_samples_split': [3, 10, 30], #'min_samples_split': [2, 3, 10, 30],
            'min_samples_leaf': [1, 5, 10],
            'max_features': [3, 4, 5, 10, 'auto'],
            'class_weight': [None, 'balanced']
        }, cv=GroupShuffleSplit(n_splits=3)),
        ensemble.RandomForestClassifier(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(svm.SVC(), {
    #        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #        #'kernel': ['linear'],
    #        'C': [1, 10, 100],
    #        'degree': [1, 2, 3],
    #        'coef0': [0, 10, 100],
    #        'tol': [0.001, 0.1, 1],
    #        'class_weight': ['balanced', None],
    #        'max_iter': [2000]
    #    }, cv=GroupShuffleSplit(n_splits=3)),
    #    svm.SVC(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
    #        'criterion': ["gini", "entropy"],
    #        'max_depth': [5, 10, None],
    #        'min_samples_split': [10, 30, 50],
    #        'class_weight': [None, 'balanced'],
    #        #'max_features': [5, None],
    #        'max_features': [5, 10, None],
    #        'min_samples_leaf': [1, 5, 10]
    #    #}, cv=5),
    #    }, cv=GroupShuffleSplit(n_splits=3)),
    #    tree.DecisionTreeClassifier(random_state=42), 'URL'),
    #ClassifierTunning(GridSearchCV(MLPClassifier(), {
    #        'hidden_layer_sizes': [5, 10, 30],
    #        'activation': ['relu'],
    #        #'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #        'solver': ['lbfgs', 'sgd', 'adam'],
    #        'alpha': [0.0001, 0.01, 0.1],
    #        'max_iter': [2000],
    #        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #        'random_state': [42]
    #    }, cv=GroupShuffleSplit(n_splits=3)),
    #    MLPClassifier(random_state=42), 'URL'),
    GroupKFoldCV(GroupShuffleSplit(n_splits=10), 'URL', cross_val_score_using_sampling)
])
result = pipeline.execute(open('data/07042020/07042020-dataset.binary.hist.arff').read())
print('Model: ' + str(result['model']))
print('Features: ' + str(result['features']))
print('X dimensions:' + str(result['X'].shape))
#print('Trainning F1: ' + str(result['score']['train_f1_macro']))
print('Test      F1: ' + str(result['score']['test_f1']))
#print('Trainning F1: %f' % (reduce(lambda x,y: x+y, result['score']['train_f1_macro']) / 10))
print('Test      F1: %f' % (reduce(lambda x,y: x+y, result['score']['test_f1']) / 10))
#print('Trainning Precision: ' + str(result['score']['train_precision_macro']))
print('Test      Precision: ' + str(result['score']['test_precision']))
#print('Trainning Recall: ' + str(result['score']['train_recall_macro']))
print('Test      Recall: ' + str(result['score']['test_recall']))
print('Test     ROC AUC: ' + str(result['score']['test_roc_auc']))
#print(' === BEST === ')
#print('Best      F1: ' + str(result['score']['best_f1']))
#print('Best      Precision: ' + str(result['score']['best_precision']))
#print('Best      Recall: ' + str(result['score']['best_recall']))
#print('Best     ROC: ' + str(result['score']['best_roc']))
#print('Best     ROC: %f' % (reduce(lambda x, y: x+y, result['score']['best_roc']) / 10))

print(' --- Features RFECV with %d features --- ' % (rfecv.n_features_))
for i in range(len(result['features'])):
    print('%s -> %d' % (result['features'][i], rfecv.ranking_[i]))

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
