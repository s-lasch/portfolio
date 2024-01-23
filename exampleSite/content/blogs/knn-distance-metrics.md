---
title: "Distance Metrics for KNN"
date: 2023-06-07T17:34:22-05:00
draft: false
author: "Steven Lasch"
tags:
  - applied math
  - data viz
  - plotly
image: /images/knn.gif
description: "Distance metrics are a crucial component of a k-nearest neighbors algorithm. They determine the distance between data points of different classes. Generally speaking, data points closer together are of the same class."
toc: true
mathjax: true
plotly: true
---

> **NOTE:** *all figures in this post were made by the author using* \\(\LaTeX\\), `numpy`, *and* `matplotlib`

We use distance formulas in \\(k\\)-NN  to determine the proximity of data points in order to make predictions or classifications based on the neighbors. There are many ways to measure similarity, along with many instances where one formula should be used over another.

<br>

### **Euclidean Distance**

The first—and most common—distance formula is the **Euclidean distance**. 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/3bf29a5d9bb4f88a9ab2e78445a0e44d45f36189/exampleSite/content/blogs/knn-distance/euclidean_distance.svg" 
           width="60%"/>
</p>

This is calculated by finding the difference between elements in list \\(x\\) with elements in list \\(y\\), calculating the sum of those differences, and taking the square root of the sum. This finds the **linear distance** between two points. 

Euclidean distance is a straightforward measure of spatial similarity, making it suitable for many applications. It is used when the features have a clear geometric interpretation, and scale differences between features are not a concern as scale difference is a major drawback with this metric.

<br>

### **Manhattan Distance**

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/ff3a80b82866d2110c137d68d335cc8df6142c8b/exampleSite/content/blogs/knn-distance/manhattan_distance.svg" 
           width="60%"/>
</p>

This distance formula is different from Euclidean distance because it does not measure the magnitude nor the angle of the line connecting two points. In certain instances, knowing the magnitude of the line between two points is necessary in a \\(k\\)-NN problem. 

When classifying a point, a shorter distance between that point and another point of a different class often indicates a *higher similarity* between the points. Consequently, the point is more likely to belong to the class that is closer to it.

You can see the difference in Euclidean distance and Manhattan distance more clearly in the image below. The formula on the right resembles the distance from one street to another in a city grid, hence the name “Manhattan” distance.

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/35b26df6a03b462736347811353eefa9edf314d5/exampleSite/content/blogs/knn-distance/euclid_manhat_distance.svg" 
           alt=""/>
</p>

The Manhattan distance can be particularly useful in datasets or scenarios where the features have different units of measurement that are all independent of each other. It captures the total discrepancy along each feature dimension without assuming any specific relationship between them.

When calculating the similarity or distance between two houses, using the Euclidean distance would implicitly assume that the features contribute equally to the overall similarity with a straight line connecting them. However, in reality, the differences in square footage, distance to a local school, number of bedrooms, etc. might not have equal importance.

<br>

### **Minkowski Distance**

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/e91ba6288d7acfe2ea5af2b1f6501b185f07f3e9/exampleSite/content/blogs/knn-distance/minkowski_distance.svg" 
           width="60%"/>
</p>

This distance formula is unique in that it includes both Euclidean and Manhattan distances as special cases, when \\(p=2\\) and \\(p=1\\), respectively. Using this distance formula allows us to control a single variable, \\(p\\), to get either formula. 

Note that [`sklearn.neighbors.KNeighborsClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#:~:text=%3D2%2C-,metric%3D%27minkowski%27,-%2C%20metric_params%3D) function uses Minkowski distance as the default metric, most likely because of its versatility. Refer to [`scipy.spatial.distance()`](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) for a complete list of distance metrics.

In general, a higher value of \\(p\\) can give more importance to larger differences between feature values, while a lower value of \\(p\\) can prioritize individual feature differences.

<br>

### **Cosine Similarity**

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/8a0c2efb79e280e357967627c45afbfdcbf50e4e/exampleSite/content/blogs/knn-distance/cosine_similarity.svg" width="50%/>
</p>

If you’ve taken a linear algebra class, you’ve definitely seen this formula before. This equation calculates \\(\cos{(\theta)}\\) , where \\(\theta\\) represents the angle between two non-zero feature vectors. It involves taking the dot product of two vectors in the numerator, then dividing it by the length of each vector.

In a linear algebra textbook, you might see a similar equation that looks like this:

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/508cb4ede63bfe74b09733c95f8bbf59b6e9d576/exampleSite/content/blogs/knn-distance/lin_alg_cosine_sim.svg" />
</p>

This is the same formula, where \\(\vec{x}\\) and \\(\vec{y}\\)  represent two feature vectors, and \\(\|\|\vec{x}\|\|\\) and \\(\|\|\vec{y}\|\|\\) are the lengths of each vector. This formula measures the similarity of two vectors, and an output range of \\([-1, \ 1]\\). Vectors where \\(\cos{(\theta)} \approxeq -1\\), have *exact dissimilarity*, \\(\cos{(\theta)} \approxeq 0\\) (orthogonal vectors), have *no correlation*, and \\(\cos{(\theta)} \approxeq 1\\) have the *exact similarity*. This can be seen graphically, as below:

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/35b26df6a03b462736347811353eefa9edf314d5/exampleSite/content/blogs/knn-distance/cos_similarity_graph.svg" 
           alt=""/>
</p>

It is important to remember that the range of cosine is between \\(-1\\) and \\(1\\). A value of \\(-1\\) indicates exact dissimilarity, \\(0\\) indicates no similarity, and \\(1\\) indicates exact similarity. In this example, the cosine similarity between the two vectors is \\(-0.7642\\), which indicates \\(\vec{x}\\) and \\(\vec{y}\\) are quite dissimilar.

You can see the differences in each cosine similarity value below. In the first one, since the two vectors are perpendicular (or orthogonal if you know your linear algebra) they are the least similar. As 2D vectors, the components in both dimensions are maximally different, with correlating dimensions having opposite signs.

In the middle graph, since the two vectors are multiples of each other, and they have the same direction, they fall on the same line in 2D space. This means they are essentially the same vector, just with a different magnitude. 

This is apparent because we have \\(2\vec{x} = \vec{y}\\) in the middle graph. Finally, in the third graph, these two vectors are as dissimilar as they can get, with a similarity of \\(-1\\). Like in the middle graph, these vectors are indeed multiples of each other since we have \\(-1\vec{x} = \vec{y}\\). The key difference is that the negative magnitude changes the direction of \\(\vec{y}\\). 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/35b26df6a03b462736347811353eefa9edf314d5/exampleSite/content/blogs/knn-distance/all_cos_sims.svg" 
           alt=""/>
</p>

<br>

### **Hamming Distance** 

<p align="center">
      <img src="https://raw.githubusercontent.com/s-lasch/portfolio/35b26df6a03b462736347811353eefa9edf314d5/exampleSite/content/blogs/knn-distance/hamming_distance.svg" 
           width="60%"/>
</p>

If we wanted to classify a binary output, this is the metric we want to use. The function \\(\delta\\) is the [Kronecker delta function](https://www.wikiwand.com/en/Kronecker_delta#:~:text=In%20mathematics%2C%20the%20Kronecker%20delta%20(named%20after%20Leopold%20Kronecker)%20is%20a%20function%20of%20two%20variables%2C%20usually%20just%20non%2Dnegative%20integers.%20The%20function%20is%201%20if%20the%20variables%20are%20equal%2C%20and%200%20otherwise%3A), which returns \\(1\\), or True if \\(x_i = y_i\\), and \\(0\\), or False if \\(x_i \neq y_i\\). It measures the number of positions at which two vectors differ. It is commonly used in various fields, including biology, for comparing sequences such as DNA molecules to identify the positions where they differ.
