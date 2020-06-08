library(PMCMR)

recall_internal <- read.table('../../recall-internal.csv', header=T, sep=',')

recall_classifiers <- c('browserbite.nn.internal.k3',
                        'crosscheck.randomforest.internal.k3',
                        'browserninja1.randomforest.internal.k3',
                        'browserninja2.randomforest.internal.k3',
                        'browserninja3.randomforest.internal.k3')

print('============== Shapiro recall-Internal ===================')
png('recall-internal.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(recall_internal[,'browserbite.nn.internal.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(recall_internal[,'crosscheck.randomforest.internal.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(recall_internal[,'browserninja1.randomforest.internal.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(recall_internal[,'browserninja2.randomforest.internal.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(recall_internal[,'browserninja3.randomforest.internal.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(recall_internal[,'browserbite.nn.internal.k3'])
shapiro.test(recall_internal[,'crosscheck.randomforest.internal.k3'])
shapiro.test(recall_internal[,'browserninja1.randomforest.internal.k3'])
shapiro.test(recall_internal[,'browserninja2.randomforest.internal.k3'])
shapiro.test(recall_internal[,'browserninja3.randomforest.internal.k3'])

print('============== ANOVA recall-Internal ==============')
recall <- c(recall_internal[, 'browserbite.nn.internal.k3'],
            recall_internal[, 'crosscheck.randomforest.internal.k3'],
            recall_internal[, 'browserninja1.randomforest.internal.k3'],
            recall_internal[, 'browserninja2.randomforest.internal.k3'],
            recall_internal[, 'browserninja3.randomforest.internal.k3'])
classifiers <- rep(recall_classifiers, each=10)
data <- data.frame(classifiers, recall)

anova <- aov(data$recall ~ data$classifiers)
print(anova)
summary(anova)

print('======== Tukey HSD ======')
TukeyHSD(x=anova)
