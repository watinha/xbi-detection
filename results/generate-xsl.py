import pandas as pd

with pd.ExcelWriter('./analysis.xlsx') as writer:
    for class_attr in ['external', 'internal']:
        for metric in ['precision', 'recall', 'fscore', 'roc']:
            df = pd.read_csv('./%s-%s.csv' % (metric, class_attr))

            crosscheck = [ col for col in list(df.columns) if col.startswith('crosscheck') ]
            browserbite = [ col for col in list(df.columns) if col.startswith('browserbite') ]
            browserninja1 = [ col for col in list(df.columns) if col.startswith('browserninja1') ]
            browserninja2 = [ col for col in list(df.columns) if col.startswith('browserninja2') ]

            sheetname = '%s-%s' % (class_attr, metric)
            print('writing %s for %s XBIs...' % (metric, class_attr))

            df.to_excel(writer, columns=(crosscheck+browserbite+browserninja1+browserninja2),
                    sheet_name=sheetname)
