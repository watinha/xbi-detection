library(PMCMR)

precision_external <- read.table('../../precision-external.csv', header=T, sep=',')

precision_classifiers <- c('browserbite.randomforest.external.k3',
                        'crosscheck.randomforest.external.k3',
                        'browserninja1.randomforest.external.k3',
                        'browserninja2.randomforest.external.k3',
                        'browserninja3.randomforest.external.k3')

print('============== Shapiro Precision-External ===================')
png('precision-external.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(precision_external[,'browserbite.randomforest.external.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(precision_external[,'crosscheck.randomforest.external.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(precision_external[,'browserninja1.randomforest.external.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(precision_external[,'browserninja2.randomforest.external.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(precision_external[,'browserninja3.randomforest.external.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(precision_external[,'browserbite.randomforest.external.k3'])
shapiro.test(precision_external[,'crosscheck.randomforest.external.k3'])
shapiro.test(precision_external[,'browserninja1.randomforest.external.k3'])
shapiro.test(precision_external[,'browserninja2.randomforest.external.k3'])
shapiro.test(precision_external[,'browserninja3.randomforest.external.k3'])

print('============== ANOVA Precision-External ==============')
precision <- c(precision_external[, 'browserbite.randomforest.external.k3'],
               precision_external[, 'crosscheck.randomforest.external.k3'],
               precision_external[, 'browserninja1.randomforest.external.k3'],
               precision_external[, 'browserninja2.randomforest.external.k3'],
               precision_external[, 'browserninja3.randomforest.external.k3'])
classifiers <- rep(precision_classifiers, each=10)
data <- data.frame(classifiers, precision)

anova <- aov(data$precision ~ data$classifiers)
print(anova)
summary(anova)

print('======== Tukey HSD ======')
TukeyHSD(x=anova)
