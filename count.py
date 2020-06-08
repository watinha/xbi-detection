import arff
import pandas as pd

df = pd.read_csv('data/07042020/level2.csv')
complete = arff.load(open('data/07042020/07042020-dataset.arff'))

attributes = [attr[0] for attr in complete['attributes']]
urls = sorted(set([ row[attributes.index('URL')] for row in complete['data']]))

data = []

for url in urls:
    df_url = df.loc[df['URL'] == url]
    df_iphone8plus = df_url.loc[df_url['targetPlatform'] == 'iOS 13.1 - Safari -- iOS - iPhone 8 Plus']
    df_iphone8plus2 = df_url.loc[df_url['targetPlatform'] == 'iOS 12.1 - Safari -- iOS - iPhone 8 Plus']
    df_iphonese = df_url.loc[df_url['targetPlatform'] == 'iOS 13.1 - Safari -- iOS - iPhoneSE']
    df_iphonese2 = df_url.loc[df_url['targetPlatform'] == 'iOS 12.1 - Safari -- iOS - iPhoneSE']
    df_motog4 = df_url.loc[df_url['targetPlatform'] == 'Android null - Chrome -- Android - MotoG4']

    df_iphone8plus_internal = df_iphone8plus.loc[df_iphone8plus['internal'] == 2]
    df_iphone8plus_external = df_iphone8plus.loc[df_iphone8plus['external'] == 2]
    df_iphone8plus_internal2 = df_iphone8plus2.loc[df_iphone8plus2['internal'] == 2]
    df_iphone8plus_external2 = df_iphone8plus2.loc[df_iphone8plus2['external'] == 2]
    df_iphonese_internal = df_iphonese.loc[df_iphonese['internal'] == 2]
    df_iphonese_external = df_iphonese.loc[df_iphonese['external'] == 2]
    df_iphonese_internal2 = df_iphonese2.loc[df_iphonese2['internal'] == 2]
    df_iphonese_external2 = df_iphonese2.loc[df_iphonese2['external'] == 2]
    df_motog4_internal = df_motog4.loc[df_motog4['internal'] == 2]
    df_motog4_external = df_motog4.loc[df_motog4['external'] == 2]

    row = [url,
           df_motog4_external.shape[0],
           df_motog4_internal.shape[0],
           df_iphonese_external.shape[0] + df_iphonese_external2.shape[0],
           df_iphonese_internal.shape[0] + df_iphonese_internal2.shape[0],
           df_iphone8plus_external.shape[0] + df_iphone8plus_external2.shape[0],
           df_iphone8plus_internal.shape[0] + df_iphone8plus_internal2.shape[0]]
    data.append(row)

new_df = pd.DataFrame(data=data, columns=['URL', 'motog4-external', 'motog4-internal',
                                          'iphonese-external', 'iphonese-internal',
                                          'iphone8plus-external', 'iphone8plus-internal'])
new_df.to_csv('data/07042020/xbis_classified.csv')
