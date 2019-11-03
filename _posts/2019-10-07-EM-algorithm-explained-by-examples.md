---
layout: post
title: "EM algorithm explained by examples"
date: 2019-10-07
tags: maths theory
image: galaxies.jpg
---

In this post I want to give an intuitive explanation of <span id="highlight">Expectation-Maximization (EM) algorithm</span>. Let's start by looking at
two well-known clustering algorithms, namely:
* <span class="marker_underline">k-means</span>
* <span class="marker_underline">Gaussian Mixture Model (GMM)</span>

We will dissect them from EM perspective and then connect this knowledge to the theory.

## k-means as EM
Let's first take a look at k-means algorithm.
<div><img src="/assets/img/EM/k-means0.png" width="35%" style="float:left"><br/>
The goal of k-means algorithm is to separate the given data $x_1, ... , x_n$ into $k$ clusters by iteratively re-fitting means of those clusters. Assume we have 10 unlabeled data points that we want to separate into 2 clusters.
</div>
<strong>k-means algorithm</strong>:
<div style="clear:both"/>
<div><img src="/assets/img/EM/k-means1.png" width="35%" style="float:left"><br/>
1) Randomly pick cluster centers $ \mu_1, ... , \mu_k$. In our case $k=2$, so we have 2 centers (green and red crosses correspond to their means).
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
<b>M-step</b>: update cluster centers. New cluster center's coordinates are an average of the coordinates of points that belong to this cluster:
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

Now let's look at Gaussian Mixture Model algorithm.

<div><img src="/assets/img/EM/EM0.png" width="55%" style="float:right">
As in k-means, we start with unlabeled set of points that we want to cluster. Let's assume that we have 10 data points (illustrated on the right) and $k=2$, so we have 2 clusters.
</div>
<div style="clear:both"/>
<br/><strong>Gaussian Mixture Model algorithm:</strong>


<div><img src="/assets/img/EM/EM1.png" width="55%" style="float:right">
1) Randomly intialize $k$ Gaussian distributions.
<br/>
<br/>In our example we put 2 Gaussians - $N(\mu_1, \sigma_1)$ and $N(\mu_2, \sigma_2)$. Probability density functions (PDFs) of these Gaussians are shown in blue and orange colors on the right plot.
</div>
<div style="clear:both"/>

2) Repeat until convergence:
<div><img src="/assets/img/EM/EM2.png" width="55%" style="float:right">
<strong>E-step</strong>: assign each point to one of the $k$ Gaussian distributions. We assign each point to the Gaussian under which this data point is most likely to be observed. So we compute something called <strong>responsibility</strong> for every data point:
$$ r_{ij} = \frac{\pi_j N(x_i | \mu_j, \sigma_j)}{\sum_{l}\pi_l N(x_i | \mu_l, \sigma_l)} $$

$\pi_j$ are weighs associated with each Gaussian (at the moment of initialization we assume that $\pi_j=1/k$). $r_{ik}$ is the probability that point $x_i$ comes from Gaussian $k$. In our case, each data point $x_i$ will have two responsibilities - $r_{i1}$ and $r_{i2}$ that sum up to 1. Each point is assigned to a cluster with a higher corresponding responsibility.

</div>
<div style="clear:both"/>


<div><img src="/assets/img/EM/EM3.png" width="55%" style="float:right"><br/>
<strong>M-step</strong>: update parameters of the Gaussian distributions. We will use responsibilities from E-step to get new parameters values:
$$ \pi_j = \frac{1}{N}\sum_{i}r_{ij}$$
$$ \mu_j = \frac{ \sum_{i}r_{ij}x_i }{ \sum_{i}r_{ij} }$$
$$ \sigma_j^2 = \frac{ \sum_{i}r_{ij}(x_i-\mu_j)^2 }{ \sum_{i}r_{ij} }$$
</div>
<div style="clear:both"/>

## Maths behind Expectation-Maximization
We can see that for both k-means and GMM algorithms, the clustering process consists of two steps:
* assigning points to a cluster (E-step),
* changing cluster's parameters based on points that were assigned to this cluster (M-step).

Although initial clusters are randomly initialized and are not necessarily a good fit for the data, after a couple of EM updates clusters explain the data much better. Let's understand EM updates from theoretical perspective.

### Goal of EM
<span class="marker_underline">The goal of Expectation-Maximization</span> <span class="marker_underline">algorithm is to find parameter</span> <span class="marker_underline">values that maximize</span> <span class="marker_underline"> the probability </span> <span class="marker_underline"> of observing the </span> <span class="marker_underline"> given data</span>. Take a look at these two images:
<div><img src="/assets/img/EM/bad_params.png" width="50%" style="float:left">
<img src="/assets/img/EM/good_params.png" width="50%" style="float:right">
<div/>

Image at the left shows two Gaussians that don't explain the data well and, hence, probability of observing the current data under that model is low. In contrast, image on the right shows a much better model.
<br/>
We want to maximize the probability of the given data under our model:
$$ p(X | \theta) = p(x_1 | \theta) \cdot p(x_2 | \theta) \cdot ... \cdot p(x_n | \theta) \rightarrow \max_{\theta}$$


Since it is easier to operate with sum rather than with product, we can take $\log$ of $p(X | \theta)$. The logarithm is an increasing function so the maximization result will be equivalent. Now our goal is <strong>maximizing data log-likelihood</strong>:
$$ l(\theta) =  \log p(X | \theta) = \sum_{i} \log p(x_i | \theta) \rightarrow \max_{\theta}$$

Since our model consists of $k$ sub-models or "clusters", we can break the data log-likelihood into sum of probabilities of observing data points under each of the "clusters". In other words, let's marginalize log-likelihood with respect to a hidden variable $z$ that will correspond to the cluster number:

$$ l(\theta) = \sum_{i} \log p(x_i | \theta) = \sum_{i} \log \sum_{z_i} p(x_i, z_i | \theta) \rightarrow \max_{\theta} $$

In this case $p(x_i, z_i | \theta)$ is the probability of observing data point $x_i$ under cluster $z_i \in \{1, ... , k\}$ given current model parameters $\theta$.
<br/><br/>
But maximizing $l(\theta)$ is difficult because of the summation inside the $\log$. The solution that EM algorithm proposes is to iteratively construct and optimize a lower bound for $l(\theta)$. Let's see how we can find a lower bound:

$$ \sum_{i} \log \sum_{z_i} p(x_i, z_i | \theta) = \sum_{i} \log \sum_{z_i} Q_i(z_i) \frac{p(x_i, z_i | \theta)}{Q_i(z_i)}
\geqslant \sum_{i} \sum_{z_i} Q_i(z_i) \log \frac{p(x_i, z_i | \theta)}{Q_i(z_i)} $$

$Q_i(z_i)$ is any probability distribution. The last transition was performed using Jensen's inequality. To get the tight lower bound, we want inequality to turn into equality, according to Jensen's inequality this happens when:
$$ \frac{p(x_i, z_i | \theta)}{Q_i(z_i)} = const $$  

So we pick $Q_i(z_i) \propto	p(x_i, z_i | \theta)$:
$$ Q_i(z_i) =  \frac{p(x_i, z_i | \theta)}{ \sum_{j} p(x_i, z_j | \theta) } = \frac{p(x_i, z_i | \theta)}{ p(x_i | \theta) }  = p(z_i | x_i, \theta)$$

<span id="highlight">That's how we get EM steps:</span> <br/> <br/>
1. <strong>E-step</strong>: evaluate $Q_i(z_i) = p(z_i | x_i, \theta)$ - probabilities of each point belonging to a cluster $z_i$.
This allows us to construct a lower bound $$\mathcal{L}(Q_i, \theta) = \sum_{i} \sum_{z_i} Q_i(z_i) \log \frac{p(x_i, z_i | \theta)}{Q_i(z_i)} = \sum_{i} \sum_{z_i} p(z_i | x_i, \theta) \log \frac{p(x_i, z_i | \theta)}{p(z_i | x_i, \theta)}  $$

2. <strong>M-step</strong>: update parameters $\theta$ to $\theta^{new}$
$$\theta^{new} = \arg \max_{\theta^{new}} \sum_{i} \sum_{z_i} p(z_i | x_i, \theta) \log \frac{p(x_i, z_i | \theta^{new})}{p(z_i | x_i, \theta)}  $$

Basically at this step we're fixing cluster assignments of each point $Q_i(z_i)$ at the old parameter values $\theta$, but allow joint distribution $p(x_i, z_i | \theta^{new})$ to change with respect to model parameters. Thus, we modify our model in such way that it best fits the cluster assignments of the data points.
