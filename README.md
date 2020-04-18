# 20Apr17_SimpleAiyagari1999
 Code for a simple Aiyagari 1999 model.



## Setup

Typical household:
$$
\begin{align}
\max & \mathbb{E}_{0} \sum^{\infty}_{t=0} \beta^{t} \frac{c_{t}^{1-\mu}}{1-\mu}, \mu>0 \\
\text{s.t. }& \begin{cases}
	c_{t} + a_{t+1} \leq w l_{t} + a_{t} \\
	\log{l_{t}} = \rho \log{l_{t}} + \sigma (1-\rho^{2})^{1/2}\varepsilon_{t} \\
	\varepsilon_{t} \sim N(0,1) \\
	a_{t+1} \geq 0
\end{cases}
\end{align}
$$
Firm:
$$
Y = F(K,L) = K^{\theta}L^{1-\theta}
$$
Baseline parameters:

| Parameters | Values |
| ---------- | ------ |
| $\beta$    | 0.96   |
| $\mu$      | 3       |
| $\theta$| 0.08|
|$\rho$ | 0.6|
|$\sigma$|0.4|