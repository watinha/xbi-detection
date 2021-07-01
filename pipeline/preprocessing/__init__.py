from sklearn.preprocessing import StandardScaler

class Preprocessor ():
    def execute (self, argument):
        X = argument['X']
        scaler = StandardScaler()
        X_new = scaler.fit_transform(X)
        argument['X'] = X_new
        return argument
