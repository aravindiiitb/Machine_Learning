# Setting working directory
setwd("~/personal/r-practicing/iris_data_analysis")

require(caTools)
# install.packages("caTools")
library("nnet")
library("caret")
# install.packages("caret")

# Read the data
data = read.csv(file="data.csv",header = TRUE, sep = ",")
data
# Print the statstics of the data
summary(data)

# Define train control for k fold cross validation
# train_control <- trainControl(method="cv", number=10)

# Train/Test split of the data
# set.seed(101)
# sample = sample.split(data$class, SplitRatio = .8)
# train = subset(data, sample == TRUE)
# test = subset(data, sample == FALSE)

set.seed(123)
ind_train <- lapply(split(seq(1:nrow(data)), data$class), function(x) sample(x, floor(.6*length(x))))
ind_test <- mapply(function(x,y) setdiff(x,y), x = split(seq(1:nrow(data)), data$class), y = ind_train)

# head(train)
# head(test[,1:4])

# Multinomial Logistic Regression
model = multinom(class ~ ., data = data[unlist(ind_train),])
summary(model)

output = predict(model, data = data[unlist(ind_test),][,1:4])
final <- cbind(output, actuals=data[unlist(ind_test),][,5:5])


# Scatter plot of the data
png(file= "scatter_plot.jpg")
plot(data[,1:4])


# Give the chart file a name.
png(file = "line_chart.jpg")

# Plot the bar chart. 
plot(data[,1:1],type = "o")

png(file = "histogram.png")
hist(data[,1:1],xlab = "Weight",col = "yellow",border = "blue")
hist(data[,1:4])

# Save the file.
dev.off()






# Reference: https://github.com/bensadeghi/R-demos/blob/master/CRAN-R/2_Iris_Predictive_Analysis.R
data(iris)
head(iris)

# Randomly sample iris data to generate training and testing sets
ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.7, 0.3))
train_data <- iris[ind == 1,]
test_data <- iris[ind == 2,]

###########################################
# Species classification with decision tree

# Create the decision tree. The tree is attempting to predict Species based upon sepal length, width and petal length and width
install.packages("party", dependencies = TRUE)
library(party)

my_formula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(my_formula, data = train_data)

# Print the tree
print(iris_ctree)

# Plot the tree
plot(iris_ctree)

# Plot a simplified view of the tree
plot(iris_ctree, type = "simple")

# Check the predictions with the training set (confusion matrix)
preds <- predict(iris_ctree)
table(preds, train_data$Species)

# Check the predictions with the test set
preds <- predict(iris_ctree, newdata = test_data)
cm <- table(preds, test_data$Species)
cm

# Compute accuracy of test predictions
sum(diag(cm)) / sum(cm)


###########################################
# Species classification with decision tree

# Create the Multinomial Logistic Regression. This attempting to predict Species based upon sepal length, width and petal length and width

# my_formula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
model = multinom(my_formula, data = train_data)


# Print the tree
print(model)

# Plot the tree
# plot(model)

# Plot a simplified view of the tree
# plot(iris_ctree, type = "simple")

# Check the predictions with the training set (confusion matrix)
preds <- predict(model)
cm1 <- table(preds, train_data$Species)
cm1
sum(diag(cm1)) / sum(cm1)

# Check the predictions with the test set
preds <- predict(model, newdata = test_data)
cm <- table(preds, test_data$Species)
cm

# Compute accuracy of test predictions
sum(diag(cm)) / sum(cm)

