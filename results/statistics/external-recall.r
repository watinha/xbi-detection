library(PMCMRplus)

data <- read.csv('../recall-external.csv')

approaches <- c(
    'browserbite.dt.external.k300', 'browserbite.randomforest.external.k300', 'browserbite.svm.external.k300', 'browserbite.nn.external.k300',
    'crosscheck.dt.external.k300', 'crosscheck.randomforest.external.k300', 'crosscheck.svm.external.k300', 'crosscheck.nn.external.k300',
    'browserninja1.dt.external.k300', 'browserninja1.randomforest.external.k300', 'browserninja1.svm.external.k300', 'browserninja1.nn.external.k300',
    'browserninja2.dt.external.k300', 'browserninja2.randomforest.external.k300', 'browserninja2.svm.external.k300', 'browserninja2.nn.external.k300' # best RFE result
)

for (i in approaches) {
    print(paste('--- ', i, ' ---'))
    print(shapiro.test(data[,i]))
}

mat <- data.matrix(data[, approaches])
friedman.test(mat)
#posthoc.friedman.nemenyi.test(mat)
frdAllPairsNemenyiTest(mat)
