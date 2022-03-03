import pandas as pd
import scikit_posthocs as sp

from scipy import stats


def run_tests(df):
    strategies = ['browserbite', 'crosscheck', 'browserninja1', 'browserninja2']
    algs = ['dt', 'randomforest', 'svm', 'nn']

    is_parametric = True

    for strategy in strategies:
        for alg in algs:
            r = stats.shapiro(df.loc[:, '%s-%s' % (strategy, alg)])
            print('Shapiro wilk in %s with %s - p-value=%f' % (strategy, alg, r.pvalue))
            if r.pvalue < 0.05:
                is_parametric = False

    params = df.to_numpy().tolist()
    if is_parametric:
        anova_result = stats.f_oneway(*params)
        print('One-Way Anova F=%f and p-value=%f' %
                (anova_result.statistic, anova_result.pvalue))
        tukey_result = stats.tukey_hsd(*params)
        matrix = tukey_result.pvalue.tolist()
        print('Posthoc Nemenyi pairwise comparisons test for unreplicated blocked data')
        cache = {}
        for i, row in enumerate(matrix):
            for j, pvalue in enumerate(row):
                if ('%d-%d' % (j, i)) in cache: continue
                cache['%d-%d' % (i, j)] = True
                if pvalue < 0.05:
                    mean_i = df.loc[:, df.columns[i]].mean()
                    mean_j = df.loc[:, df.columns[j]].mean()
                    bigger_index = i if mean_i > mean_j else j
                    smaller_index = i if mean_i < mean_j else j

                    print(' - %s (%f) and %s (%f) with p-value=%f' %
                            (df.columns[bigger_index], df.loc[:, df.columns[bigger_index]].mean(),
                             df.columns[smaller_index], df.loc[:, df.columns[smaller_index]].mean(),
                             pvalue))
    else:
        friedman_result = stats.friedmanchisquare(*params)
        print('Friedman test for repeated samples F=%f and p-value=%f' %
                (friedman_result.statistic, friedman_result.pvalue))
        nemeyi_result = sp.posthoc_nemenyi_friedman(df)

        print('Posthoc Nemenyi pairwise comparisons test for unreplicated blocked data')
        cache = {}
        for col in nemeyi_result.columns.tolist():
            for row in nemeyi_result.index.tolist():
                if ('%s-%s' % (row, col)) in cache: continue
                cache['%s-%s' % (col, row)] = True
                pvalue = nemeyi_result.loc[row, col]
                if pvalue < 0.05:
                    mean_i = df.loc[:, col].mean()
                    mean_j = df.loc[:, row].mean()
                    bigger_index = col if mean_i > mean_j else row
                    smaller_index = col if mean_i < mean_j else row

                    print(' - %s (%f) and %s (%f) with p-value=%f' %
                            (bigger_index, df.loc[:, bigger_index].mean(),
                             smaller_index, df.loc[:, smaller_index].mean(), pvalue))


metrics = ['precision', 'recall', 'fscore']
incompatibilities = ['external', 'internal']

for inc in incompatibilities:
    for met in metrics:
        print('''
*******************************
    %s - %s
*******************************
''' % (met, inc))
        df = pd.read_csv('../%s-%s.csv' % (met, inc), index_col=0)
        df.columns = [ col.replace('-%s' % (inc), '') for col in df.columns ]
        df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)
        run_tests(df)




