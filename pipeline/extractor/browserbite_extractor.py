import np

class BrowserbiteExtractor ():
    def __init__ (self, class_attr):
        self._class_attr = class_attr

    def execute (self, arff_dataset):
        arff_dataset['features'] = [
                'diff_height', 'diff_width', 'diff_x', 'diff_y',
                'missmatch', # 'correlation'
                'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
                'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                'target_bin1', 'target_bin2', 'target_bin3', 'target_bin4', 'target_bin5',
                'target_bin6', 'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10']
        X_t = []

        attributes = [ attr[0] for attr in arff_dataset['attributes'] ]
        data = arff_dataset['data']

        X_t.append(np.abs(np.array(data[:, attributes.index('baseHeight')], dtype='float64') -
                          np.array(data[:, attributes.index('targetHeight')], dtype='float64')))
        X_t.append(np.abs(np.array(data[:, attributes.index('baseWidth')], dtype='float64') -
                          np.array(data[:, attributes.index('targetWidth')], dtype='float64')))
        X_t.append(np.abs(np.array(data[:, attributes.index('baseX')], dtype='float64') -
                          np.array(data[:, attributes.index('targetX')], dtype='float64')))
        X_t.append(np.abs(np.array(data[:, attributes.index('baseY')], dtype='float64') -
                          np.array(data[:, attributes.index('targetY')], dtype='float64')))
        X_t.append(np.ones(len(data)))
        X_t.append(np.array(data[:, attributes.index('base_bin1')]))
        X_t.append(np.array(data[:, attributes.index('base_bin2')]))
        X_t.append(np.array(data[:, attributes.index('base_bin3')]))
        X_t.append(np.array(data[:, attributes.index('base_bin4')]))
        X_t.append(np.array(data[:, attributes.index('base_bin5')]))
        X_t.append(np.array(data[:, attributes.index('base_bin6')]))
        X_t.append(np.array(data[:, attributes.index('base_bin7')]))
        X_t.append(np.array(data[:, attributes.index('base_bin8')]))
        X_t.append(np.array(data[:, attributes.index('base_bin9')]))
        X_t.append(np.array(data[:, attributes.index('base_bin10')]))
        X_t.append(np.array(data[:, attributes.index('target_bin1')]))
        X_t.append(np.array(data[:, attributes.index('target_bin2')]))
        X_t.append(np.array(data[:, attributes.index('target_bin3')]))
        X_t.append(np.array(data[:, attributes.index('target_bin4')]))
        X_t.append(np.array(data[:, attributes.index('target_bin5')]))
        X_t.append(np.array(data[:, attributes.index('target_bin6')]))
        X_t.append(np.array(data[:, attributes.index('target_bin7')]))
        X_t.append(np.array(data[:, attributes.index('target_bin8')]))
        X_t.append(np.array(data[:, attributes.index('target_bin9')]))
        X_t.append(np.array(data[:, attributes.index('target_bin10')]))

        arff_dataset['X'] = np.array(X_t, dtype='float64').T
        arff_dataset['y'] = np.array(data[:, attributes.index('xbi')], dtype='float64')
        return arff_dataset
