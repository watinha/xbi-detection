library(PMCMRplus)

data <- read.csv('../recall-internal.csv')

approaches <- c(
    'browserbite.dt.internal.k300', 'browserbite.randomforest.internal.k300', 'browserbite.svm.internal.k300', 'browserbite.nn.internal.k300',
    'crosscheck.dt.internal.k300', 'crosscheck.randomforest.internal.k300', 'crosscheck.svm.internal.k300', 'crosscheck.nn.internal.k300',
    'browserninja1.dt.internal.k300', 'browserninja1.randomforest.internal.k300', 'browserninja1.svm.internal.k300', 'browserninja1.nn.internal.k300',
    'browserninja2.dt.internal.k300', 'browserninja2.randomforest.internal.k300', 'browserninja2.svm.internal.k300', 'browserninja2.nn.internal.k300' # best RFE result
)

for (i in approaches) {
    print(paste('--- ', i, ' ---'))
    print(shapiro.test(data[,i]))
}

mat <- data.matrix(data[, approaches])
#posthoc.friedman.nemenyi.test(mat)
frdAllPairsNemenyiTest(mat)
