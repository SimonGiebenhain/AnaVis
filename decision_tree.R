library(RWeka)
MsgId <- as.factor(c(1:17))
TimeZone <- as.factor(c('US', 'US', 'EU', 'AS', 'AS','AS', 'EU', 'US', 'US', 'AS','US', 'EU', 'EU', 'AS', 'US', 'AS', 'EU'))
GeoLocation <- as.factor(c('US','US','US','EU','AS','AS','AS','EU','AS','EU','EU','EU','US','EU', 'US', 'AS', 'AS'))
SuspiciousSubject <- as.factor(c('No','No','No','No', 'Yes','Yes','Yes','No','Yes','Yes','Yes','No','Yes','No', 'Yes', 'No', 'No'))
SuspiciousBody <- as.factor(c('Yes', 'No', 'Yes','Yes','Yes','No','No','Yes','Yes','Yes','No','No','Yes','No', 'Yes', 'No', 'Yes'))
Spam <- as.factor(c('No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No', 'tba', 'tba', 'tba'))
input <-data.frame(MsgId,TimeZone, GeoLocation, SuspiciousSubject, SuspiciousBody, Spam)
train <- input[c(1:14),]
pred <- input[c(15:17),]
tree <- J48(Spam~., train)
plot(tree)
predict(tree, pred)
