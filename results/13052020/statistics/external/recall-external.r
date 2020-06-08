library(PMCMR)

recall_external <- read.table('../../recall-external.csv', header=T, sep=',')

recall_classifiers <- c('browserbite.randomforest.external.k3',
                        'crosscheck.randomforest.external.k3',
                        'browserninja1.randomforest.external.k3',
                        'browserninja2.randomforest.external.k3',
                        'browserninja3.randomforest.external.k3')

print('============== Shapiro recall-External ===================')
png('recall-external.png', height=300, width=1200, units='px')
par(mfrow=c(1,5))
plot(density(recall_external[,'browserbite.randomforest.external.k3']), main='Browserbite RF', xlim=c(0, 1))
plot(density(recall_external[,'crosscheck.randomforest.external.k3']), main='CrossCheck RF', xlim=c(0, 1))
plot(density(recall_external[,'browserninja1.randomforest.external.k3']), main='BrowserNinja 1 RF', xlim=c(0, 1))
plot(density(recall_external[,'browserninja2.randomforest.external.k3']), main='BrowserNinja 2 RF', xlim=c(0, 1))
plot(density(recall_external[,'browserninja3.randomforest.external.k3']), main='BrowserNinja 3 RF', xlim=c(0, 1))
dev.off()

shapiro.test(recall_external[,'browserbite.randomforest.external.k3'])
shapiro.test(recall_external[,'crosscheck.randomforest.external.k3'])
shapiro.test(recall_external[,'browserninja1.randomforest.external.k3'])
shapiro.test(recall_external[,'browserninja2.randomforest.external.k3'])
shapiro.test(recall_external[,'browserninja3.randomforest.external.k3'])

print('============== ANOVA recall-External ==============')
recall <- c(recall_external[, 'browserbite.randomforest.external.k3'],
            recall_external[, 'crosscheck.randomforest.external.k3'],
            recall_external[, 'browserninja1.randomforest.external.k3'],
            recall_external[, 'browserninja2.randomforest.external.k3'],
            recall_external[, 'browserninja3.randomforest.external.k3'])
classifiers <- rep(recall_classifiers, each=10)
data <- data.frame(classifiers, recall)

anova <- aov(data$recall ~ data$classifiers)
print(anova)
summary(anova)

print('======== Tukey HSD ======')
TukeyHSD(x=anova)
