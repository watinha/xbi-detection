from sklearn import preprocessing

class Preprocessor ():
    def execute (self, argument):
        X = argument['X']
        X_new = preprocessing.scale(X)
        argument['X'] = X_new
        return argument
