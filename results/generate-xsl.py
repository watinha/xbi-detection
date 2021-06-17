import pandas as pd

class_attr = 'external'

with pd.ExcelWriter('./analysis.xlsx') as writer:
    for class_attr in ['external', 'internal']:
        for metric in ['precision', 'recall', 'fscore', 'roc']:
            df = pd.read_csv('./%s-%s.csv' % (metric, class_attr))
            sheetname = '%s-%s' % (class_attr, metric)
            df.to_excel(writer, sheet_name=sheetname)
