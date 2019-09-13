from sklearn import tree
from sklearn.model_selection import GridSearchCV

from pipeline import Pipeline
from pipeline.loader.arff_loader import ArffLoader
from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.classifier.classifier_tunning import ClassifierTunning

features = ['childsNumber', 'textLength']

pipeline = Pipeline([
    ArffLoader(),
    XBIExtractor(features, 'Result'),
    ClassifierTunning(GridSearchCV(tree.DecisionTreeClassifier(), {
        'criterion': ["gini", "entropy"],
        'max_depth': [10, 60, 100, None],
        'min_samples_split': [2, 30, 100],
        'class_weight': [None, 'balanced']
    }, cv=2),
    tree.DecisionTreeClassifier(random_state=42))
])
result = pipeline.execute(open('data/dataset-040919.arff').read())
print(result)
