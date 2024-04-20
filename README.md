# PAR-NP
Pseudo-Auto-Regressive Neural Process Model. I've recently become very interested in the [Neural Process](https://arxiv.org/abs/1807.01622)(NP) family of models and their variants. At the time of writing, the most performant NP models are:

- [Auto-Regressive Conditional Neural Processes](https://arxiv.org/abs/2303.14468)(AR-CNP)
- [Diffusion-Augmented Contional Neural Processes](https://arxiv.org/abs/2311.09848) (DANP)

I made this repository because I'm interested in exploring some of the puzzling things about both of these models (for a review of how NPs work, see [here](https://xyzml.medium.com/neural-processes-explained-2bd9b225412b) and [here](https://yanndubs.github.io/Neural-Process-Family/text/Intro.html) but it's probably best to read the papers in question). 



## High-level goals/Mathematical motivations
At a high level, to explain the goals of this repository we only need to know the following (apologies for the weird superscripts and subscripts - making markdown maths behave is hard):

Let:
- $\mathcal{X}=\mathbb{R}^{d_x}$,  $\mathcal{Y}=\mathbb{R}^{d_y}$ denote the input and output spaces for prediction
- $(x,y) \in \mathcal{X} \times \mathcal{Y}$ denote an input–output pair.
- $\mathcal{S}=\cup_{N=0}^{\infty}(\mathcal{X},\mathcal{Y})^{N}$ be a collection of all finite data sets, which includes the empty set ∅, the data set containing no data points.
  
We denote:
- a context set with $\mathcal{D}^{C} \in \mathcal{S}$, where $|\mathcal{D}^{C}|$ $= N^{C}$
- a target set with $\mathcal{D}^{T} \in \mathcal{S}$, where $|\mathcal{D}^{T}|$ $= N^{T}$

Let $X^{C} \in \mathbb{R}^{N^{C} \times d_x}$, $Y^{C} \in \mathbb{R}^{N^{C} \times d_y}$ be the inputs and corresponding outputs of $\mathcal{D}_{C}$ with:
- $X^{T} \in \mathbb{R}^{N^{T} \times d_{x}}$
- $Y^{T} \in \mathbb{R}^{N^{T} \times d_{y}}$

defined analogously. We denote a single task a $\eta= (\mathcal{D}^{C},\mathcal{D}^{T})= ((X^{C},Y^{C}),(X^{T},Y^{T}))$. Let $\mathcal{P}(\mathcal{X})$ denote the collection of stochastic processes on $\mathcal{X}$. 


In the DANP paper, a de-noising diffusion process is integrated into the standard CNP architecture. 
