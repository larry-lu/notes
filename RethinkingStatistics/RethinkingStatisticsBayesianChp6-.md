# Rethinking Statistics

## Chp6 Overfitting, Regularization, and Information Criteria

Two families to approach the trade off between over-/under-fitting:

1. Regularizing prior
2. Information criteria

AIC uses MAP estimates instead of the entire posterior distribution, and the prior is flat. The Bayesian Information Criterion (BIC) also requires flat priors and MAP estimates, although it's not really an "information criterion". Bayesian criteria like DIC and WAIC provide almost exactly the same results as AIC.

### 6.1 The problem with parameters

$R^2$ - "variance explained", is defined as:

$R^2 = \frac{var(outcome)-var(residuals)}{var(outcome)}=1-\frac{var(residuals)}{var(outcome)}$

Easy to compute, but poor choice for model selection.

#### More parameters always improve fit

1st order:

```python
m_6_1 = smf.ols('brain ~ mass', data=d).fit()
1 - m_6_1.resid.var()/d.brain.var()
# m_6_1.summary() check the value for R-squared
```

$R^2$: 0.4901580479490838

2nd - 6th order:

```python
m_6_2 = smf.ols('brain ~ mass + I(mass**2)', data=d).fit()
m_6_3 = smf.ols('brain ~ mass + I(mass**2) + I(mass**3)', data=d).fit()
m_6_4 = smf.ols('brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4)', data=d).fit()
m_6_5 = smf.ols('brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4) + I(mass**5)', data=d).fit()
m_6_6 = smf.ols('brain ~ mass + I(mass**2) + I(mass**3) + I(mass**4) + I(mass**5) + I(mass**6)', data=d).fit()
for model in [m_6_2, m_6_3, m_6_4, m_6_5, m_6_6]:
    print('R2: ', 1 - model.resid.var()/d.brain.var())
```

R2 for 2nd order model:  0.5359967130300407
R2 for 3rd order model:  0.6797736224399112
R2 for 4th order model:  0.8144339074066689
R2 for 5th order model:  0.9888540172279255
R2 for 6th order model:  0.9914536203218768

The increasing over-fitting is obvious.

It's possible to consider model fitting as a form of data compression. The compression sums the data in a simpler form, although information is lost about the sample.

For a "perfectly" over-fitted model, there is no compression. The model just encoded the data in a different form using parameters instead. Therefore, we learn nothing about the data through the model. This view of model selection is known as minimum description length (MDL).

#### Too few parameters hurts, too

Under-fitted models are less sensitive compared to regular models, and over-fitted models are too sensitive.

![underfitting and overfitting](rethinking/under_over_fit.png)

### Information theory and model performance

The target/criterion for model evaluation used in information theory: the out-of-sample deviance.

1. joint probability, not average probability, is the right way to judge model accuracy
2. establish a measurement scale for distance from perfect accuracy
3. deviance as an approximation of relative distance from perfect accuracy
4. deviance out-of-sample

The definition of the target depends on:

1. cost-benefit analysis.
2. accuracy in context.

Joint probability is the multiplication of probability from each event in the sequence. It appears in Bayes' theorem as the likelihood.

>Information: The reduction in uncertainty derived from learning an outcome.

Information theory is about how much the uncertainty reduced by learning an outcome.

Three intuitive desiderata:

1. The measure of uncertainty should be continuous.
2. The measure of uncertainty should increase as the number of possible events increases.
3. The measure of uncertainty should be additive.

The function that satisfies these desiderata: **Information Entropy**. If there are $n$ different possible events and each event $i$ has probability $p_i$, then we call the list of probabilities p, then the measure of uncertainty we seek is:

$H(p) = -E \log(p_i) = -\sum_{i=1}^{n}p_i \log(p_i)$

Or

>the uncertainty contained in a probability distribution is the average log-probability of an event.

E.g. to compute the information entropy for the weather, and we suppose the true probability of rain and shine are $p_1=0.3$ and $p_2=0.7$. Then:

$H(p)=-(p_1 \log(p_1) + p_2 \log(p_2)) \approx 0.61$

```python
p = (0.3, 0.7)
-sum(p * np.log(p))
#0.6108643020548935
```

If we live in a location where $p_1=0.01$ and $p_2=0.99$, then the uncertainty $H(p)=-(p_1 \log(p_1) + p_2 \log(p_2)) \approx 0.06$. The uncertainty has decreased. 

Adding an event to the dimensionality, sun, rain, and snow: $p_1=0.3$, $p_2=0.3$, and $p_3=0.7$. Then the entropy $H(p) \approx 0.82$.

Note: changing the base of the function does not affect the inference. Also, when $p_i=0$, do not include $p_i\log(p_i)$ in the calculation.

`MAXENT` maximizes the entropy, so that the distribution is the most consitent with the states of knowledge.

#### From entropy to accuracy

Divergence, describes the distance from the model to the target using information entropy. It's often known as the *Kullback-Leibler divergence* or K--L divergence. 

>Divergence: The additional uncertainty induced by using probabilities from one distribution to describe another distribution.

For example, if true distribution: $p_1=0.3$, $p_2=0.7$, and we believe it's $q_1=0.25$ and $q_2=0.75$, then the additional uncertainty is calculated as:

$D_{KL}(p, q)=\sum_{i}p_i(\log(p_i)-\log(q_i))=\sum_{i}p_i\log(\frac{p_i}{q_i})$

If p = q, then $D_{KL}(p,q)=0$, no extra uncertainty is introduced.

**Cross entropy**: introduced when we use distribution $q$ to predict events from distribution $p$. The divergence is defined as the *additional* entropy introduced by using $q$:

$D_{KL}(p, q) = H(p,q) - H(p) = -\sum_i p_i \log(q_i) - (- \sum_i p_i \log(p_i)) = -\sum_i p_i (\log(q_i) - \log(p_i))$

**Note**: knowing which is the target matters. $H(p, q)$ does not equal $H(q,p)$.

A good informative example (p180):

>In general, H(p, q) is not equal to H(q, p). The direction matters, when computing divergence. Understanding why this is true is of some value, so here’s a contrived teaching example.
Suppose we get in a rocket and head to Mars. But we have no control over our landing spot, once we reach Mars. Let’s try to predict whether we land in water or on dry land, using the Earth to provide a probability distribution q to approximate the actual distribution on Mars, p. For the Earth, $q = {0.7, 0.3}$, for probability of water and land, respectively. Mars is very dry, but let’s say for the sake of the example that there is 1% surface water, so $p = {0.01, 0.99}$. If we count the ice caps, that’s not too big a lie. Now compute the divergence going from Earth to Mars. It turns out to be
$D_{E→M} = D_{KL}(p, q) = 1.14$. That’s the additional uncertainty induced by using the Earth to predict the Martian landing spot. Now consider going back the other direction. The numbers in p and q stay the same, but we swap their roles, and now $D_{M→E} = D_{KL} (q, p) = 2.62$. The divergence is more than double in this direction. This result seems to defy comprehension. How can the distance from Earth to Mars be shorter than the distance from Mars to Earth?
Divergence behaves this way as a feature, not a bug. There really is more additional uncertainty induced by using Mars to predict Earth than by using Earth to predict Mars. The reason is that, going from Mars to Earth, Mars has so little water on its surface that we will be very very surprised when we most likely land in water on Earth. In contrast, Earth has good amounts of both water and dry land. So when we use the Earth to predict Mars, we expect both water and land, to some extent, even though we do expect more water than land. So we won’t be nearly as surprised when we inevitably arrive on Martian dry land, because 30% of Earth is dry land.

#### From divergence to deviance

Problem arises when in real life, the true value of $p$ is almost never known. Therefore, we cannot calculate the $D_{KL}$ to compare the models. But we can compare the relative divergences of different models. Since the parts related to $p$ are the same for any model to calculate the $D_{KL}$, we can just use the other terms (log-probabilities) for comparison.

To approximate the relative value of log-probability $E(\log(q_i))$, we can calculate the deviance:

$D(q) = -2 \sum_i \log(q_i)$

#### From deviance to out-of-sample

Deviance is similar to $R^2$ in the sense that the value improves as model gets more complex. Thus, using out-of-sample deviance as testing dataset to evaluate the models.

1. Suppose there's a training sample of size $N$
2. Fit a model to the training sample, and compute the deviance on the training sample as $D_{train}$
3. Suppose another sample of size $N$ as the test sample
4. Compute the deviance on the test sample $D_{test}$ using the MAP estimates from step 2

### Regularization

One way to prevent over-fitting is to assign a regularizing prior which is skeptical. The way to do it: assign a small standard deviation, so the model is more skeptical about values further away from the mean ($\mu=0$ if all variables are scaled).

Regularization on priors needs to be tuned: if too skeptical then underfitting.

### Information criteria

#### AIC

Akaike Information Criterion:

$AIC = D_{train} + 2p$

where $p$ is the number of free parameters to be estimated in the model.

AIC is only reliable when:

1. The priors are flat or overwhelmed by the likelihood.
2. The posterior distribution is approximately multivariate Gaussian.
3. The sample size $N$ is much greater than the number of parameters $k$. 

#### DIC

Deviance Information Criterion:

A version of AIC that is aware of informative priors, also assumes multivariate Gaussian posterior distribution.

$DIC = \bar{D} + (\bar{D} - \hat{D}) = \bar{D}+p_D$

where $(\bar{D}$ represents the average of $D$, the posterior distribution of deviance. The $\hat{D})$ represents the deviance calculated at the posterior mean. The $p_D$ is also often called a penalty term.

#### WAIC

Widely Applicable Information Criterion (WAIC) does not require a multivariate Gaussian posterior. More accurate than DIC.

WAIC is pointwise: calculates uncertainty case-by-case.

$Pr(y_i)$ as the average likelihood of observation $i$ in the training sample, and calculate the log-pointwise-predictive-density, $lppd$:

$lppd = \sum_{i=1}^{N} \log Pr(y_i)$

> the log-pointwise-predictive-density is the total across observations of the log-arithm of the average likelihood of each observation. 