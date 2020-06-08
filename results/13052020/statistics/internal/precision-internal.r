library(PMCMR)

precision_internal <- read.table('../../precision-internal.csv', header=T, sep=',')

precision_classifiers <- c('browserbite.nn.internal.k3',
                        'crosscheck.randomforest.internal.k3',
                        'browserninja1.randomforest.internal.k3',
                        'browserninja2.randomforest.internal.k3',
                        'browserninja3.randomforest.internal.k3')

print('============== Shapiro precision-Internal ===================')
png('precision-internal.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(precision_internal[,'browserbite.nn.internal.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(precision_internal[,'crosscheck.randomforest.internal.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(precision_internal[,'browserninja1.randomforest.internal.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(precision_internal[,'browserninja2.randomforest.internal.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(precision_internal[,'browserninja3.randomforest.internal.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(precision_internal[,'browserbite.nn.internal.k3'])
shapiro.test(precision_internal[,'crosscheck.randomforest.internal.k3'])
shapiro.test(precision_internal[,'browserninja1.randomforest.internal.k3'])
shapiro.test(precision_internal[,'browserninja2.randomforest.internal.k3'])
shapiro.test(precision_internal[,'browserninja3.randomforest.internal.k3'])

print('============== ANOVA precision-Internal ==============')
precision <- c(precision_internal[, 'browserbite.nn.internal.k3'],
            precision_internal[, 'crosscheck.randomforest.internal.k3'],
            precision_internal[, 'browserninja1.randomforest.internal.k3'],
            precision_internal[, 'browserninja2.randomforest.internal.k3'],
            precision_internal[, 'browserninja3.randomforest.internal.k3'])
classifiers <- rep(precision_classifiers, each=10)
data <- data.frame(classifiers, precision)

anova <- aov(data$precision ~ data$classifiers)
print(anova)
summary(anova)

print('======== Tukey HSD ======')
TukeyHSD(x=anova)
