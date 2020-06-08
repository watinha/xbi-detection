library(PMCMR)

roc_external <- read.table('../../roc-external.csv', header=T, sep=',')

roc_classifiers <- c('browserbite.nn.external.k3',
                     'crosscheck.dt.external.k3',
                     'browserninja1.nn.external.k3',
                     'browserninja2.randomforest.external.k3',
                     'browserninja3.randomforest.external.k3')

print('============== Shapiro ROC-External ===================')
png('roc-external.png', height=200, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(roc_external[,'browserbite.nn.external.k3']), main='Browserbite MLP', xlim=c(0, 1))
plot(density(roc_external[,'crosscheck.dt.external.k3']), main='CrossCheck DT', xlim=c(0, 1))
plot(density(roc_external[,'browserninja1.nn.external.k3']), main='BrowserNinja 1 MLP', xlim=c(0, 1))
plot(density(roc_external[,'browserninja2.randomforest.external.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(roc_external[,'browserninja3.randomforest.external.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(roc_external[,'browserbite.nn.external.k3'])
shapiro.test(roc_external[,'crosscheck.dt.external.k3'])
shapiro.test(roc_external[,'browserninja1.nn.external.k3'])
shapiro.test(roc_external[,'browserninja2.randomforest.external.k3'])
shapiro.test(roc_external[,'browserninja3.randomforest.external.k3'])

print('============== Friedman ROC-External ==============')
mat <- data.matrix(roc_external[, roc_classifiers])
friedman.test(mat)
print(' -- post hoc analysis')
posthoc.friedman.nemenyi.test(mat)
