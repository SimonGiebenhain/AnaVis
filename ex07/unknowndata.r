library(readr)
dataset <- read_csv("Downloads/dataset.csv", col_names = FALSE)
data <- dataset[, 2:14]
data_norm <- scale(data)
minPoints <- 3
epsilon <- 3
noise <- c()
for (i in 1:nrow(data_norm)) {
  counter <- 0
  for (j in 1:nrow(data_norm)) {
    if (dist(rbind(data_norm[i, ], data_norm[j, ])) < epsilon) {
      counter <- counter + 1
    }
  }
  if (counter < minPoints) {
    noise[length(noise) + 1] <- i
  }
}
data_norm <- data_norm[-noise, ]
c_ward <- hclust(dist(data_norm), method = "ward.D2")
ward_cut <- cutree(c_ward, 3)
pca <- prcomp(data[], center=TRUE, scale.=TRUE)
dim_red <- predict(pca, newdata = data[])
col <- c()
j <- 1
for (i in 1:178) {
  if (i %in% noise) {
    col[i] <- 0
  } else {
    col[i] <- ward_cut[j]
    j <- j + 1
  }
}
library('plotly')
plot_ly(data.frame(dim_red), x = dim_red[, 1], y = dim_red[, 2], z=dim_red[, 3], color = col)

