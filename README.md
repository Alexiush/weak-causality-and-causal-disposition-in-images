This repository contains source code for the models used for the experiments in study "Discover and Explore Weak Causality and Causal Disposition in Images for Smart Manufacturing Tasks (Streltsov, Terziyan, Vitko)" [To be published].
We also provide Jupyter Notebooks that show how to use and interpret this models.

# CA-CNNs and related work
Many ML models even when demonstrate great results make their predictions based on correlation rather than causation. Such behavior is not beneficial as produced models are hard to interpret. 
There are already many alternative architectures that enforce use of causation in models, but they often rely on extensive usage of external knowledge. In some cases it is just very hard to get needed amount of external knowledge from human experts,
but sometimes it is pretty much impossible, like in work with digital media where there is no direct mapping from bits to objects on scene, sounds, etc.

## Causal disposition
In [Discovering Causal Signals in Images (Lopez-Paz et al.)](https://arxiv.org/abs/1605.08179) it was shown how the weak causal signals in image datasets can be found via "causal dispositions". Causal disposition is some property or ability of an object
that can be a cause of the other object's presence. To discover the causal signal between some objects A and B one can count the number of images where B dissapear when A is removed and vice versa and compare them receiveng an asymetry between those object with
the higher count showing the cause.

## CA-CNN
In [Causality-Aware Convolutional Neural Networks for Advanced Image Classification and Generation (Terziyan, Vitko)](https://www.sciencedirect.com/science/article/pii/S1877050922023237) two new methods to calculate the causal asymetry in joint probability fashion 
were introduced:

$P(F^i|F^j) = \frac{P(F^i, F^j)}{P(F^j)}$  

`Max`: $P(F^i|F^j) = \frac{(max F^i) * (max F^j)}{\sum_{l,r=1}^{n}F^j}$ 

`Lehmer`: $P(F^i|F^j) = \frac{LM_\alpha(F^i \times F^j)}{LM_\alpha(F^j)}$  

Also the CA-CNN (Causality Aware CNN) architecture was proposed that is of the model that calculates the matrix of causal asymetries based on feature maps extracted by CNN and uses it for "causal aggregation" before passing it to the classifier.
The proposed way to causally aggregate the features was to simply concatenate the features and causal estimates.


## Prior empirical studies
In [Exploiting causality signals in medical images: A pilot study with empirical results (Carloni, Colantonio)](https://www.sciencedirect.com/science/article/pii/S0957417424002987) CA-CNN was implemented and tested on practice. The study confirmed that there is a 
notable uplifts in model performance and interpretability after going causal. New method for causal aggregation was presented to address the problem with heterogeneity of concatenated features, it was named `Mulcat` as it multiplied the features by the weight
based on asymetries favoring the feature and only then concatenated them.

# Our results
In our study we have tested on practice the behaviour of various CA-CNN configurations evaluated on wide set of different image classification datasets to address the most important questions about practical usage of CA-CNNs and to answer the questions "Why CA-CNN?"
and "How to CA-CNN?". 

Experiments show that `Lehmer` is mostly better than `Max`, however the difference is not very big and `Max` can be favorable as it is much lighter in computation. However, `Lehmer` is also shown to be more robust and noise tolerant. 

Original causality aggregation method (`Concat`) and `Mulcat` were compared showing that the latter is better and is a great base for future research.

We found that computing causality maps with pre-trained CNNs significantly enhances classification performance of CA-CNNs and, especially, training efficiency.

Special saliency maps based on causal maps were introduced as a form of model interpretation and knowledge representation.
![Special saliency maps or "Causal shadows"](https://i.imgur.com/yybyOJ8.png)
