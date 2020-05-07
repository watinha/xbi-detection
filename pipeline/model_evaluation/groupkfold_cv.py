class GroupKFoldCV():

    def __init__(self, folds, group_attr, cross_val_score):
        self._folds = folds
        self._group_attr = group_attr
        self._cross_val_score = cross_val_score

    def execute(self, args):
        attributes = [ attribute[0] for attribute in args['attributes'] ]
        groups = list(args['data'][:,attributes.index(self._group_attr)])

        args['score'] = self._cross_val_score(args['model'], args['X'], args['y'].astype('int'),
                cv=self._folds, groups=groups, scoring=['f1', 'precision', 'recall'])
        return args
