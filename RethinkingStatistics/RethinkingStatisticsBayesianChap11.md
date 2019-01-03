# Chapter 11 Monsters and mixtures

## Ordered categorical outcomes

When the outcome is a series of ordered values (ratings of Amazon).

The caveat: the differences in values are not necessarily equal (1 to 2 not necessarily equal to 5 to 6).

Solution: cumulative link function

### Example, moral intuition

Describing the distribution of discrete ordered values:

![cumulative proportion](rethinking/cumulative_proportion.png)

To redescribe the log-cumulative odds, a series of intercept is needed, each one on the log-cumulative-odds scale and stands for the cumulative probability of each outcome. The log-cumulative-odds that a response value $y_i$ is equal to or less than possible outcome value $k$ is:

$\log \frac{\Pr(y_i \le k)}{1 - \Pr(y_i \le k)} = \alpha_k$

Where $\alpha_k$ is an intercept unique to each possible outcome value $k$. The $\alpha_k$ can be calculated as:

```python
resp_lco = (trolley_df.response
                      .value_counts()
                      .sort_index()
                      .cumsum()
                      .iloc[:-1]
                      .div(trolley_df.shape[0])
                      .apply(lambda p: np.log(p / (1. - p))))
print(resp_lco)
#[-1.91609116 -1.26660559 -0.718634    0.24778573  0.88986365  1.76938091]
```

The log cumulative odds for the 7th response is $\inf$ and not shown.

![log cumulative odds](rethinking/log_cumulative_odds.png)

The posterior distribution of these intercepts can be achieved through calculation of likelihood $\Pr(y_i) = k$. The likelihood of $k$ can be obtained through subtraction:

$p_k = \Pr(y_i = k) = \Pr(y_i \le k) - \Pr(y_i \le k-1)$

Using the above formula, we can reversely calculate all likelihood by subtraction.

```python
class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])

with pm.Model() as m11_1:
    a = pm.Normal(
        'a', 0., 10.,
        transform=Ordered(),
        shape=6, testval=np.arange(6) - 2.5)

    pa = pm.math.sigmoid(a)

    p_cum = tt.concatenate([[0.], pa, [1.]])
    p = p_cum[1:] - p_cum[:-1]

    resp_obs = pm.Categorical(
        'resp_obs', p,
        observed=trolley_df.response - 1)

with m11_1:
    map_11_1 = pm.find_MAP()
```

posterior distribution of intercepts:

```python
map_11_1['a']
#array([-1.9160707 , -1.26658298, -0.71862013,  0.24778795,  0.88986631, 1.76937289])

with m11_1:
    trace_11_1 = pm.sample(1000, tune=1000)
pm.summary(trace_11_1, varnames=['a'], alpha=.11).round(2)
```

||mean | sd |mc_error|hpd_5.5|hpd_94.5|n_eff |Rhat|
|-----|----:|---:|-------:|------:|-------:|-----:|---:|
|a__0 |-1.92|0.03|       0|  -1.96|   -1.87|2117.4|   1|
|a__1 |-1.27|0.02|       0|  -1.30|   -1.23|2527.0|   1|
|a__2 |-0.72|0.02|       0|  -0.75|   -0.68|2350.5|   1|
|a__3 | 0.25|0.02|       0|   0.21|    0.28|2069.5|   1|
|a__4 | 0.89|0.02|       0|   0.85|    0.93|2149.8|   1|
|a__5 | 1.77|0.03|       0|   1.72|    1.82|2628.5|   1|

logistic function to get cumulative probabilities:

```python
sp.special.expit(map_11_1['a'])
#array([ 0.12830038,  0.21984275,  0.32769691,  0.56163196,  0.70886258, 0.85437967])
```

### Adding predictor variables

Adding a predictor $x$ to the model by defining a linear model $\phi_i = \beta x_i$. Then each cumulative logit becomes:

$\log \frac{\Pr(y_i \le k)}{1 - \Pr(y_i \le k)} = \alpha_k - \phi_i$
$\phi_i = \beta x_i$

>Why linear model $\phi$ subtracted from each intercept? Because if we decrease the log-cumulative-odds of every outcome value k below the maximum, this necessarily shifts probability mass upwards towards higher outcome values.

If we calculate MAP estimate from m11.1:

```python
def ordered_logistic_proba(a):
    pa = sp.special.expit(a)
    p_cum = np.concatenate(([0.], pa, [1.]))

    return p_cum[1:] - p_cum[:-1]

ordered_logistic_proba(trace_11_1['a'].mean(axis=0))
#array([ 0.12817034,  0.09156755,  0.10796181,  0.23405232,  0.1472369, 0.14552313,  0.14548796])
```

Average outcome:

```python
(ordered_logistic_proba(trace_11_1['a'].mean(axis=0)) \
     * (1 + np.arange(7))).sum()
#4.1991390877567065
```

Now we subtract 0.5 from each:

```python
ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5)
#array([ 0.08186793,  0.06402364,  0.08229003,  0.20921751,  0.15899878, 0.1844197 ,  0.21918241])
```

Smaller values got smaller and larger values got larger. Expected value:

```python
(ordered_logistic_proba(trace_11_1['a'].mean(axis=0) - 0.5) \
     * (1 + np.arange(7))).sum()
#4.7294442978950277
```
