import pandas as pd, arff

pd.set_option('display.max_colwidth', None)

def load_arff(filename):
    arff_data = {}
    with open(filename, 'r') as f:
        arff_data = arff.load(f)

    attributes = [ attr[0] for attr in arff_data['attributes'] ]
    return pd.DataFrame(arff_data['data'], columns=attributes)

def get_alexa_urls():
    with open('./url_list_alexa.txt', 'r') as urls_file:
        urls = urls_file.readlines()

    return [ url[:-1] for url in urls ]

df = load_arff('./data/19112021/dataset.classified.hist.img.external.arff')
urls = df['URL'].unique().tolist()
alexa_urls = get_alexa_urls()
students, alexa = [], []
urls_map = {}
result = []

for url in urls:
    df['external'] = pd.to_numeric(df['external'], downcast='signed')
    df['internal'] = pd.to_numeric(df['internal'], downcast='signed')
    url_data = df.loc[df['URL'] == url].loc[df['targetPlatform'] != 'null'].groupby('targetPlatform')
    counts, *others = url_data.count()['id'].tolist()
    internal = url_data.sum()['internal'].tolist()
    external = url_data.sum()['external'].tolist()

    t = url[25:].split('/')[0]
    if (t.isnumeric()):
        urls_map[url] = alexa_urls[int(t)]
        alexa.append([urls_map[url], counts] + list(sum(zip(external, internal), ())))
    else:
        base = 'http://wwatana.be/mobile/'
        end = url[25:].replace('.notranslate', '')
        urls_map[url] = '%s%s' % (base, end)
        students.append([urls_map[url], counts] + list(sum(zip(external, internal), ())))

alexa.sort(key=lambda el: el[0])
students.sort(key=lambda el: el[0])

result_df = pd.DataFrame(students, columns=['URL', 'elements',
    'external', 'internal', 'external', 'internal', 'external', 'internal'])
print(result_df.to_latex(index=False, column_format='|l|r|r|r|r|r|r|r|'))
result_df = pd.DataFrame(alexa, columns=['URL', 'elements',
    'external', 'internal', 'external', 'internal', 'external', 'internal'])
print(result_df.to_latex(index=False, column_format='|l|r|r|r|r|r|r|r|'))
