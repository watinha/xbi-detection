import random, arff, sys, pandas as pd

from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from functools import reduce

from config import get_extractor, get_classifier, get_sampler
from pipeline import Pipeline
from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.loader.arff_loader import ArffLoader

random.seed(42)

def main(extractor_name, classifier_name, class_attr, sampler_name, n_splits):
    (extractor, features, nfeatures, max_features) = get_extractor(
            extractor_name, class_attr)
    gridsearch = get_classifier(classifier_name, nfeatures, max_features)
    sampler = get_sampler(sampler_name)

    approach = '%s-%s-%s' % (extractor_name, classifier_name, class_attr)
    print('running --- %s...' % (approach))

    extractor_pipeline = Pipeline([
        ArffLoader(), XBIExtractor(features, class_attr), extractor ])
    data = extractor_pipeline.execute(open(
        './xbi-detection/data/19112021/dataset.classified.hist.img.%s.arff' % (class_attr)).read())
    X, y, attributes, features = data['X'], data['y'], [ attr[0] for attr in data['attributes'] ], data['features']
    groups = list(data['data'][:, attributes.index('URL')])

    print('data extracted...')
    rankings, fscore, precision, recall, roc, train_fscore = [], [], [], [], [], []
    cv = GroupShuffleSplit(n_splits=n_splits, random_state=42)
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = [i for i in groups if groups.index(i) in train_index]

        X_samp, y_samp = sampler.fit_resample(X_train, y_train)
        groups_samp = [
                groups_train[X_train.tolist().index(row)] for row in X_samp.tolist() ]

        print('Model trainning with: X (%s)' % (str(X_samp.shape)))
        gridsearch.fit(X_samp, y_samp, groups=groups_samp)
        print('Model trained with fscore %s, and params %s ' % (str(gridsearch.best_score_), str(gridsearch.best_params_)))

        selector = gridsearch.best_estimator_.named_steps['selector']
        rankings.append(selector.get_support(indices=False))

        if (classifier_name == 'svm'):
            model = CalibratedClassifierCV(gridsearch.best_estimator_)
            model.fit(X_samp, y_samp)
        else:
            model = gridsearch

        y_pred = model.predict_proba(X_train)
        probability = y_pred[:, list(
            gridsearch.best_estimator_.named_steps['classifier'].classes_).index(1)]

        precision2, recall2, threasholds = metrics.precision_recall_curve(
                y_train, probability)
        best_f = 0
        threashold = 0
        for i in range(len(precision2)):
            new_fscore = (2 * precision2[i] * recall2[i]) / (precision2[i] + recall2[i])
            if new_fscore > best_f:
                best_f = new_fscore
                threshold = threasholds[i]

        print('Model training F-Score with selected threshold: %f' % (metrics.f1_score(y_train, [ 0 if prob < threshold else 1 for prob in probability])))

        train_fscore.append(metrics.f1_score(y_train, [ 0 if prob < threshold else 1 for prob in probability]))
        y_pred = model.predict_proba(X_test)
        probability = y_pred[:, list(
            gridsearch.best_estimator_.named_steps['classifier'].classes_).index(1)]

        y_threashold = [ 0 if y < threshold else 1 for y in probability]
        print('Model tested with F-Score: %f' % (metrics.f1_score(y_test, y_threashold)))
        fscore.append(metrics.f1_score(y_test, y_threashold))
        precision.append(metrics.precision_score(y_test, y_threashold))
        recall.append(metrics.recall_score(y_test, y_threashold))
        roc.append(metrics.roc_auc_score(y_test, y_threashold))


    print('Features: ' + str(features))
    print('X dimensions:' + str(X.shape))
    print('Test     ROC: %f' % (reduce(lambda x,y: x+y, roc) / n_splits))
    print('Test      F1: %f' % (reduce(lambda x,y: x+y, fscore) / n_splits))
    print('Test      F1: ' + str(fscore))
    print('Train     F1: ' + str(train_fscore))
    print('Test      Precision: ' + str(precision))
    print('Test      Recall: ' + str(recall))

    try:
        fscore_csv = pd.read_csv('results/fscore-%s.csv' % (class_attr), index_col=0)
        precision_csv = pd.read_csv(
                'results/precision-%s.csv' % (class_attr), index_col=0)
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
        features_csv = pd.DataFrame(columns=features)

    features_len = features_csv.shape[1]
    print(features)
    print(features_len)
    print(rankings)

    if extractor_name == 'browserninja2':
        for i in range(len(rankings)):
            features_csv.loc[
                    '%s-%d' % (classifier_name, (i + features_len)), :] = rankings[i]
        features_csv.to_csv('results/features-%s.csv' % (class_attr))


if __name__ == '__main__':
    assert len(sys.argv) == 5, 'The script requires 4 parameters: feature extractor (browserbite|crosscheck|browserninja1|browserninja2), classifier (randomforest|svm|dt|nn), type of xbi (internal|external) and sampler strategy (none|tomek|near|repeated|rule|random)'

    class_attr = sys.argv[3]
    extractor_name = sys.argv[1]
    classifier_name = sys.argv[2]
    sampler_name = sys.argv[4]
    n_splits = 10

    main(class_attr, extractor_name, classifier_name, sampler_name, n_splits)
