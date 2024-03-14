

The likelihood

$$
\begin{gathered}
P(y_{obs,i},y_{err,i},\mathbf{x}_{obs,i},\mathbf{x}_{err,i}|\boldsymbol{\theta})= \\
\int P(y_{obs,i},\mathbf{x}_{obs,i}|y_{err,i},\mathbf{x}_{err,i},y,\mathbf{x}) \\
P(y_{err,i},\mathbf{x}_{err,i}|y,\mathbf{x})\:P(y,\mathbf{x}|\boldsymbol{\theta})\:dy\:d\mathbf{x} \\
\approx \sum_{x, y\ \in\ \mathrm{GHQ}} w(\mathbf{x}, y) P(y_{err,i},\mathbf{x}_{err,i}|y,\mathbf{x})\:P(y,\mathbf{x}|\boldsymbol{\theta}) \\
= \sum_{x, y\ \in\ \mathrm{GHQ}} w(\mathbf{x}, y) P(\mathbf{x}_{err,i}|\mathbf{x})P(y_{err,i}|y)\:P(y|\mathbf{x}, \boldsymbol{\theta}) P(\mathbf{x}) \\
= \sum_{x, y\ \in\ \mathrm{GHQ}} w(\mathbf{x}, y) P(\mathbf{x}, \mathbf{x}_{err,i})P(y_{err,i}|y)\:P(y|\mathbf{x}, \boldsymbol{\theta})
\end{gathered}
$$

We can precalculate $w_1 = P(\mathbf{x}_i, \mathbf{x}_{err,i})$, $w_2 = P(y_{err,i}|y_i)$.

Thus, we have 

$$
P(y_{obs,i},y_{err,i},\mathbf{x}_{obs,i},\mathbf{x}_{err,i}|\boldsymbol{\theta}) 
\approx \sum_{x, y\ \in\ \mathrm{GHQ}} w(\mathbf{x}, y) w_1 w_2 \:P(y|\mathbf{x}, \boldsymbol{\theta})
$$


When we apply pre-transform in NF, which transform 
$$
x, x_{err} \rightarrow u, v
$$




The fitted NF just give $p(u, v)$, so $p(x, x_{err}) = p(u, v) |\partial (u, v)/ (x, x_{err})|$