library(PMCMR)

fscore_external <- read.table('../../fscore-external.csv', header=T, sep=',')

fscore_classifiers <- c('browserbite.randomforest.external.k3',
                        'crosscheck.randomforest.external.k3',
                        'browserninja1.randomforest.external.k3',
                        'browserninja2.randomforest.external.k3',
                        'browserninja3.randomforest.external.k3')

print('============== Shapiro FsCore-External ===================')
png('fscore-external.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(fscore_external[,'browserbite.randomforest.external.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(fscore_external[,'crosscheck.randomforest.external.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(fscore_external[,'browserninja1.randomforest.external.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(fscore_external[,'browserninja2.randomforest.external.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(fscore_external[,'browserninja3.randomforest.external.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(fscore_external[,'browserbite.randomforest.external.k3'])
shapiro.test(fscore_external[,'crosscheck.randomforest.external.k3'])
shapiro.test(fscore_external[,'browserninja1.randomforest.external.k3'])
shapiro.test(fscore_external[,'browserninja2.randomforest.external.k3'])
shapiro.test(fscore_external[,'browserninja3.randomforest.external.k3'])

print('============== Friedman FScore-External ==============')
mat <- data.matrix(fscore_external[, fscore_classifiers])
friedman.test(mat)
print(' -- post hoc analysis')
posthoc.friedman.nemenyi.test(mat)
