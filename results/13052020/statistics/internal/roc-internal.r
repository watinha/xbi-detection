library(PMCMR)

roc_internal <- read.table('../../roc-internal.csv', header=T, sep=',')

roc_classifiers <- c('browserbite.randomforest.internal.k3',
                        'crosscheck.dt.internal.k3',
                        'browserninja1.randomforest.internal.k3',
                        'browserninja2.nn.internal.k3',
                        'browserninja3.randomforest.internal.k3')

print('============== Shapiro ROC-External ===================')
png('ROC-internal.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(roc_internal[,'browserbite.randomforest.internal.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(roc_internal[,'crosscheck.dt.internal.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(roc_internal[,'browserninja1.randomforest.internal.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(roc_internal[,'browserninja2.nn.internal.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(roc_internal[,'browserninja3.randomforest.internal.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(roc_internal[,'browserbite.randomforest.internal.k3'])
shapiro.test(roc_internal[,'crosscheck.dt.internal.k3'])
shapiro.test(roc_internal[,'browserninja1.randomforest.internal.k3'])
shapiro.test(roc_internal[,'browserninja2.nn.internal.k3'])
shapiro.test(roc_internal[,'browserninja3.randomforest.internal.k3'])

print('============== Friedman ROC-Internal ==============')
mat <- data.matrix(roc_internal[, roc_classifiers])
friedman.test(mat)
print(' -- post hoc analysis')
posthoc.friedman.nemenyi.test(mat)
