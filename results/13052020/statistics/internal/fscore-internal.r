library(PMCMR)

fscore_internal <- read.table('../../fscore-internal.csv', header=T, sep=',')

fscore_classifiers <- c('browserbite.nn.internal.k3',
                        'crosscheck.randomforest.internal.k3',
                        'browserninja1.randomforest.internal.k3',
                        'browserninja2.randomforest.internal.k3',
                        'browserninja3.randomforest.internal.k3')

print('============== Shapiro FScore-Internal ===================')
png('FScore-internal.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(fscore_internal[,'browserbite.nn.internal.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(fscore_internal[,'crosscheck.randomforest.internal.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(fscore_internal[,'browserninja1.randomforest.internal.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(fscore_internal[,'browserninja2.randomforest.internal.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(fscore_internal[,'browserninja3.randomforest.internal.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(fscore_internal[,'browserbite.nn.internal.k3'])
shapiro.test(fscore_internal[,'crosscheck.randomforest.internal.k3'])
shapiro.test(fscore_internal[,'browserninja1.randomforest.internal.k3'])
shapiro.test(fscore_internal[,'browserninja2.randomforest.internal.k3'])
shapiro.test(fscore_internal[,'browserninja3.randomforest.internal.k3'])

print('============== ANOVA FScore-Internal ==============')
fscore <- c(fscore_internal[, 'browserbite.nn.internal.k3'],
            fscore_internal[, 'crosscheck.randomforest.internal.k3'],
            fscore_internal[, 'browserninja1.randomforest.internal.k3'],
            fscore_internal[, 'browserninja2.randomforest.internal.k3'],
            fscore_internal[, 'browserninja3.randomforest.internal.k3'])
classifiers <- rep(fscore_classifiers, each=10)
data <- data.frame(classifiers, fscore)

anova <- aov(data$fscore ~ data$classifiers)
print(anova)
summary(anova)

print('======== Tukey HSD ======')
TukeyHSD(x=anova)
