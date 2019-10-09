---
layout: post
title: "EM algorithm explained by examples"
date: 2019-10-07
---

In this post I want to give an intuitive explanation of <span id="highlight">Expectation-Maximization (EM) algorithm</span>. Let's start by looking at
two well-known clustering algorithms, namely:
* <span class="marker_underline">k-means</span>
* <span class="marker_underline">Gaussian Mixture Model (GMM)</span>

We will dissect them from EM perspective and then connect this knowledge to the theory.

## k-means as EM
<div><img src="/assets/img/EM/k-means0.png" width="35%" style="float:left"><br/>
The goal of k-means algorithm is to separate the given data into k clusters by iteratively re-fitting means of those clusters. Assume we have 10 unlabeled data points that we want to separate into 2 clusters.
<br/>
<br/>
<br/>
<strong>k-means algorithm</strong>:
</div>
<div style="clear:both"/>
<div><img src="/assets/img/EM/k-means1.png" width="35%" style="float:left"><br/>
1) Randomly pick cluster centers $ \mu_1, ... , \mu_k$. In our case $k=2$, so we have 2 centers (green and red crosses).
<br/>
<br/>
2) Repeat until convergence:
</div>
<div style="clear:both"/>

<div><img src="/assets/img/EM/k-means2.png" width="35%" style="float:left"><br/>
<strong>E-step</strong>: assign each point $x_i$ to one of the clusters $j \in \{1, ... , k\}$. We assign each point to the cluster whose center is closer to that point:
$$ c_i = \arg\min_{j} || \mu_j - x_i ||^2$$
In our case, $c_i$ will be either 1 or 2 depending on the cluster to which point $x_i$ was assigned.
</div>
<div style="clear:both"/>


<div><img src="/assets/img/EM/k-means3.png" width="35%" style="float:left"><br/>
<b>M-step</b>: re-calculate cluster centers. New cluster center's coordinates are an average of the coordinates of points that belong to this cluster:
$$ \mu_j = \frac{ \sum_{i}x_i \cdot \mathbb{I}\{c_i=j\} }{\sum_{i} \mathbb{I}\{c_i=j\}}$$
$j \in \{1 ,.., k\}$,
$\mathbb{I}\{c_i=j\}$ is an indicator function that tells whether point $x_i$ belongs to cluster $j$:
$$ \mathbb{I}\{c_i=j\} =
\begin{cases}
  1 & \text{if }c_i=j\\    
  0 & \text{if }c_i \neq j
\end{cases} $$

</div>
<div style="clear:both"/>



## GMM as EM
