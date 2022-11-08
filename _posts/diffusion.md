---
title: 'Diffusion Models'
date: '2022-11-08T23:06:22.456Z'
---

# Derivations
This is some notes I took while learning diffusion models. Need to clean it up & turn it into a proper blog post.

$$
\newcommand{\d}[1]{\mathrm{d}#1}
$$
$q(x_0)$ is our data-distribution. It is a datapoint, $x_0$, before any forward process diffusion.
$p_\theta$ is what we're trying to learn

$$
KL(q(x_0) | p_\theta) = \mathbb{E}_{q(x_0)}\left[\log q(x_0) - \log p_\theta\right] = \mathbb{E}_{q(x_0)}[\log q(x_0)] + \mathbb{E}_{q(x_0)}\left[-\log p_\theta\right]
$$
We can't optimize the 1st part of the final expression, so all that's left is to optimize $\mathbb{E}_{q(x_0)}[-\log p_\theta]$.
$\mathbb{E}_{q(x_0)}$ is *expectation under the data distribution*. This is equivalent to sampling from the data distribution.

Expectation definition:
$$
\displaylines{
\mathbb{E}[x] = \int_x x \cdot p(x) \d x\\
\mathbb{E}[g(x)] = \int_x g(x) \cdot p(x) \d x
}
$$
We currently have: $p_\theta(x_0)$. But the reverse diffusion process depends on latents $x_1$ through $x_T$.
We need to pull these into our expression. To do that, we use *marginalization*.

What's marginalization? 
- Let $A$ be a random variable that represents the probability of it raining tomorrow.
- Let $B$ be a random variable that represents some other event. Maybe: Tom catches Jerry today. (Note how the 2 events don't necessarily have to be related). The probability of it raining tomorrow can also be expressed as "probability of it raining tomorrow & probability that Tom catches Jerry today" + "probability of it raining tomorrow & probability that Tom does not catch Jerry today "

In this case, both $A$ and $B$ can take on 2 possible values.

$$
P(A=a) = \sum_b P(A=a, B=b)
$$
If this was continuous, we would have:
$$
P(A = a) = \int P(A=a, B=b) \d b
$$
Side note, in this case, $A$ and $B$ are independent, so we can do:

$$
\begin{align*}
P(A = a) \\
&= \sum_b P(A=a,B=b) \\
&= \sum_b P(A=a) \cdot P(B=b) \\
&= P(A=a) \cdot \sum_b P(B=b) \\
&= P(A=a) \cdot 1 \\
&= P(A = a)
\end{align*}
$$

We now reach out 1st excerpt from the paper!

$$
p_\theta(\textbf{x}_0) \coloneqq \int p_\theta(\textbf{x}_{0:T})\d\textbf{x}_{1:T}
$$
We needed to "pull in" all the latent variables that $\textbf{x}_0$ depends on. To do this, we marginalized $\textbf{x}_0$ with respect to all the latent variables in the reverse diffusion process.

Note:
$$
\displaylines{
p_\theta(x_{0:T}) = p_\theta(x_0, x_1, ..., x_T)\\
\d{x}_{1:T} = \d{x}_1 ... \d{x}_T
}
$$

### Aside: Forward Diffusion
For the forward diffusion process:

$$
q(\textbf{x}_t|\textbf{x}_{t-1}) \coloneqq \mathcal{N}(\textbf{x}_t; \sqrt{1 - \beta_t}\textbf{x}_{t-1}, \beta_t\textbf{I})
$$

We scale the mean by $\sqrt{1-\beta_t}$ in order to prevent the values/variance from exploding.
**TODO**: Be more precise here. Write some tests & *verify* what is happening.

I originally thought it was b/c we want to scale the mean down to 0. Indeed, if we put in data composed of all 1s into the forward diffusion process, the end result would have a mean of 0 and a stdev of 1.

**But**, that isn't the *goal* of scaling the mean. In diffusion models, we assume our input has a mean of 0. Because if our forward diffusion process *shifts* the mean to 0, then our neural net must learn to shift the mean back. That's probably bad since our neural net uses the *same* weights for each reverse diffusion step.

### Aside: Training-Inference Compute Asymetry
Diffusion models allow you to train with much less compute, while still utilizing a ton of compute during sampling time.

### Aside: Functional Form
> Both processes have the same functional form when $\beta_t$ are small

This is saying "we need a lot of timesteps in the forward/reverse diffusion processes" in order for both forward/reverse processes to be gaussian.

Example: Imagine your dataset consists of cats & dogs. If you do a single reverse step to go from *noise* to "full cat/dog" then your reverse step can't produce a gaussian distribution of outcomes. How do you fit a gaussian to cats & dogs? You would need a multimodal distribution instead.

### ELBO
$$
\Large
\begin{align*}
\mathbb{E}_{q(x_0)}\left[-\log p_\theta(x_0)\right]\\
&=^{1.} \mathbb{E}_{q(x_0)}\left[-\log \int p_\theta(x_{0:T}) \d{x}_{1:T}\right]\\
&=^{2.} \mathbb{E}_{q(x_0)}\left[-\log \int \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}p_\theta(x_{0:T}) \d{x}_{1:T}\right]\\
&=^{3.} \mathbb{E}_{q(x_0)}\left[-\log \mathbb{E}_{q(x_{1:T|x_0})}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]  \right]\\
&\leq^{4.} \mathbb{E}_{q(x_0)}\left[ \mathbb{E}_{q(x_{1:T}|x_0)}\left[- \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]  \right]\\
\end{align*}
$$

1. Marginalization to pull-in the latent variables
2. Bring in the forward diffusion process
3. Integral => Expectation
4. $\mathbb{E}[\log X] \leq \log \mathbb{E}[X]$ using Jensen's Inequality

#### Jensen's Inequality
How do we prove $\mathbb{E}[\log X] < \log \mathbb{E}[X]$?

##### Proof By Example
`average(log(1), log(2), log(3))` $\simeq 0.59$
`log(average(1, 2, 3))` $\simeq 0.69$

##### Picture Proof
[Desmos](https://www.desmos.com/calculator/engjmlai2o)
![[Pasted image 20220812143558.png]]

---

$$
\Large
\begin{align*}
\mathbb{E}_{q(x_0)}\left[ \mathbb{E}_{q(x_{1:T}|x_0)}\left[- \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]  \right]\\
=^{1.} \mathbb{E}_{q(x_0)}\left[\mathbb{E}_{q(x_{1:T}|x_0)}\left[-\log p(x_T) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\right]\\
\end{align*}
$$
1. Use the fact that $p_\theta$ and $q$ are defined as products + log rules.

A note about $p(x_T)$:

Technically, it should be $p_\theta(x_T)$, but we assume that after enough forward diffusion steps, $p(x_T)$ is identical to the normal distribution (hence the equation earlier in the paper: $p(x_T) = \mathcal{N}(x_T;\textbf{0},\textbf{I})$).

Concretely, to calculate $\log p(x_T)$ we could:
1. Take an image from our data set, $x_0$
2. Run forward diffusion for T steps to get $T$.
3. Calculate the probability of $x_T$ appearing from the normal distribution, $\mathcal{N}$.
	- TODO: I'm guessing you would do this by calculating the probability of the normal distribution discretizing to the given image?

### Rewriting ELBO
For simplicity, we'll write:

$$
\mathbb{E}_{q(x_0)}\left[\mathbb{E}_{q(x_{1:T}|x_0)}\left[-\log p(x_T) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\right]\\
$$
as

$$
\mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\\
$$
Now we'll simplify it:

$$
\large
\begin{align*}
&\mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t\geq1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\\
&=^{1.} \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t \geq 2} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} - \log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]\\
&=^{2.} \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t \geq 2} \left[ \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} \cdot \frac{q(x_{t-1})}{q(x_t)} \right] - \log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]\\
&=^{3.} \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t \geq 2} \log \left[ \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \right] - \log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]\\
&=^{4.} \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t \geq 2} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} - \sum_{t\geq 2} \log \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} - \log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]\\
&=^{5.} \mathbb{E}_{q}\left[-\log p(x_T) - \sum_{t \geq 2} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} -  \log \frac{q(x_1|x_0)}{q(x_T|x_0)} - \log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]\\
&=^{6.} \mathbb{E}_q\left[-\log \frac{p(x_T)}{q(x_T|x_0)} - \sum_{t \geq 2} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} - \log p_\theta(x_0|x_1) \right] \\
&=^{7.} \mathbb{E}_q\left[D_{KL}(q(x_T|x_0)\Vert p(x_T) + \sum_{t \geq 2} D_{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)\right]
\end{align*}
$$
1. Extract the 1st term out of the summation. This is necessary because when we apply Bayes' Theorem to $q$, we will get a non-sensical result if we also apply it to the 1st term in the summation.
2. Apply Bayes' rule to $q(x_t|x_{t-1})$
3. We need to condition the reverse conditional probability, $q$, on $x_0$. Why? $q(x_{t-1}|x_t)$ needs to give the probability distribution of $x_{t-1}$s given $x_t$, but this might be extremely difficult if, e.g, $x_t$ has a lot of noise (which it will near the end of the diffusion process). If we know the original image, $x_0$, this process becomes easy. *This also makes the reverse conditional probability tractable. I.e., we can compute it*. 
4. Log rules.
5. Expand the 2nd summation, apply log rules to get the log of a cumulative product, cancel terms.
6. Log rules
7. Definition of KL-divergence

### Reparameterization
We can do the forward process in 1-step.
[Need help simplifying an iterative diffusion process to a 1-step process](https://math.stackexchange.com/questions/4488937/need-help-simplifying-an-iterative-diffusion-process-to-a-1-step-process)
$$
\begin{align*}
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1 - \beta_t}, \beta_tI)\\
&=^{1.} \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon\\
&=^{2.} \sqrt{a_t}x_{t-1} + \sqrt{1-a_t}\epsilon\\
&=^{3.} \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon\right) + \sqrt{1 - \alpha_t}\epsilon\\
&=^{4.} \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\epsilon + \sqrt{1 - \alpha_t}\epsilon\\
&=^{5.} \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\epsilon
\end{align*}
$$
1. We go from $\beta_t$ to $\sqrt{\beta_t}$ because $\beta_tI$ is a covariance matrix. Specifically, it's the diagonal covariance matrix, so it only has covariance values of the form $\text{COV}(X, X) = \text{VAR}(X) = \sigma^2$
2. Substitute $\beta_t$ with $a_t$
3. Substitute using $x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon$
4.  Algebra
5.  $\epsilon$ is sampled from the normal distribution $\epsilon = \mathcal{N}(0, I)$ Thus, the last 2 terms are gaussians with mean 0 and standard deviation $\sqrt{a_t}\sqrt{1-a_{t-1}}$ and $\sqrt{1-a_t}$. $\text{VAR}(X + Y) = \text{VAR}(X) + \text{VAR}(Y)$ if $X$ and $Y$ are independent. Thus, the sum of those two terms is a gaussian with mean 0 and variance $\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t\alpha_{t-1}$

### Deriving The Posterior
*A lot of the math here comes from the excellent blog post [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*

The reverse conditional probability is:
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$

We need to calculate $\tilde\mu$ and $\tilde\beta$.

$$
\newcommand{\ncolor}{\color{white}}
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
&=^{1.} q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto^{2.} \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \ncolor{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \ncolor{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \ncolor{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \ncolor{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$
1. Bayes' Rule
2.  This is the "proportional to" symbol. We're using these facts:
	1. The PDF of a gaussian is $\frac{1}{\sqrt{2\sigma^2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
	2. $e^a \cdot \frac{e^b}{e^c} = e^{a+b-c}$
		- $q(x_t|x_{t-1},x_0) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_tI)$ (definition of $q$)
			- $\mu = \sqrt{1-\beta_t}x_{t-1} = \sqrt{a_t}x_{t-1}$
			- $\sigma = \sqrt{\beta_t}$
		- $q(x_{t}|x_0) = \mathcal{N}(x_{t}; \sqrt{\bar{a}_t}x_0, (1-\bar{a}_t)I)$  (reparameterization trick)
			- $\mu = \sqrt{\bar{a}_t}x_0$
			- $\sigma = \sqrt{1-\bar{a}_t}$
		- Same idea for $q(x_{t-1}|x_0)$

Let's examine part of the PDF of the gaussian more closely.

$$
\begin{align*}
-\frac{(x-\mu)^2}{2\sigma^2}\\
&= -\frac{x^2 - 2x\mu + \mu^2}{2\sigma^2}\\
&= -\frac{1}{2\sigma^2}x^2 + \frac{x\mu}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\\
&= -\frac{1}{2}\left(\frac{x^2}{\sigma^2} - \frac{2x\mu}{\sigma^2} + \frac{\mu^2}{\sigma^2} \right)
\end{align*}
$$

From this, we know:
$$
\begin{align}
\frac{x^2}{\sigma^2} &= \color{red}\text{red}\\
\sigma^2 &= \frac{x^2}{\color{red}\text{red}}
\end{align}
$$
$$
\begin{align}
\frac{2x\mu}{\sigma^2} = \color{blue}\text{blue}\\
\mu = \frac{\sigma^2\color{blue}\text{blue}}{2x}
\end{align}
$$

One last note, $x$ is actually $x_{t-1}$ since that is the output of $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})$.
Now we can derive the mean and variance of $q(x_{t-1}|x_t,x_0)$:

$$
\begin{aligned}
\tilde{\beta}_t
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}

$$

#### Simplifying The Posterior Mean
From reparameterization we know:

$$
\begin{align*}
x_t &= \sqrt{\bar{a}_t}x_0 + \sqrt{1 - \bar{a}_t}\epsilon\\
x_0 &= \frac{1}{\sqrt{\bar{a}_t}}\left(x_t - \sqrt{1-\bar{a}_t}\epsilon\right)
\end{align*}
$$

##### Aside: Implementation
We can do a bit more algebra to derive the terms used in the OpenAI codebase:

$$
\begin{align*}
x_0 &= \frac{1}{\sqrt{\bar{a}_t}}x_t - \frac{\sqrt{1- \bar{a}_t}}{\sqrt{\bar{a}_t}}\cdot \epsilon\\
x_0 &= \frac{1}{\sqrt{\bar{a}_t}}x_t - \sqrt{\frac{1}{\bar{a}_t} - 1}\cdot \epsilon
\end{align*}
$$
---

We can use our definition of $x_0$ and substitute:
$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t

&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\mathbf{\epsilon}) \\

&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \Big)}
\end{aligned}
$$

I could not figure out the algebra for this step. But, I did verify the results numerically.
See these [tests](https://github.com/vedantroy/improved-ddpm-pytorch/blob/main/tests/equiv_tests_theory.py).

### Closed Form KL-Divergence
We want to calculate the KL-divergence efficiently.
Normally, we would need to Monte Carlo estimation.
But, since $L_{t-1}$ is comparing 2 gaussians, we can use the [closed form of the KL Divergence](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians):

$$
D_{KL}(p, q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

The authors ignore the 1st & last terms since they don't learn the variance for the reverse diffusion process.

- Question: Why bother including $\sigma_t$ in the denominator given that we're not learning? (My guess: resembles denoised score matching)

# Extensions
## DDIM
DDIM allows deterministic sampling without modifying the training process.
Let's analyze the OpenAI code for DDIM sampling:

```python
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
		# ===MARKER 1===
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
		# ===MARKER 2===
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

### Marker 1
$$
\begin{align*}
\sigma = \text{eta} \cdot \sqrt{\frac{1 - \bar{a}_{t-1}}{1 - \bar{a}_t}} \cdot \sqrt{1 - \frac{\bar{a}_t}{\bar{a}_{t - 1}}}\\
&=^{1.} \text{eta} \cdot \sqrt{\frac{1 - \bar{a}_{t-1}}{1 - \bar{a}_t}} \cdot \sqrt{1 - (1 - \beta_t)}\\
&= \text{eta} \cdot \sqrt{\frac{1 - \bar{a}_{t-1}}{1 - \bar{a}_t}} \cdot \sqrt{\beta_t}\\
&= \text{eta} \cdot \sqrt{\tilde{\beta_t}}
\end{align*}
$$

1. Utilize the facts that A. $\bar{a}_t$ is a product B. $a_t \coloneqq 1 - \beta_t$

TODO: Why isn't the OpenAI code just using square root of $\beta_t$ directly? *My guess, float64/float32 numeric stuff*.

### Marker 2
```python
mean_pred = (
    out["pred_xstart"] * th.sqrt(alpha_bar_prev)
    + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
)
# Skip a few lines
sample = mean_pred + nonzero_mask * sigma * noise
```

$$
x_{t-1} = \sqrt{\bar{a}_{t-1}}x_0 + \sqrt{1 - \bar{a}_{t-1} - \sigma_t^2}\epsilon_\theta + \sigma_t\epsilon
$$

Where $\epsilon_\theta$ is the random noise predicted by the network and $\epsilon$ is random noise we generate.
Note, when $\text{eta} = 0$ , then this simplifies to:

$$
x_{t-1} = \sqrt{\bar{a}_{t-1}}x_0 + \sqrt{1 - \bar{a}_{t-1}}\epsilon_\theta
$$
which is a deterministic sampling process! In essence, instead of sampling from the predicted posterior mean/variance, we just use the reparameterization trick and stop 1-step earlier.