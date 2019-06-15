# Brancher: An Object-Oriented Variational Probabilistic Programming Library

Brancher allows design and train differentiable Bayesian models using stochastic variational inference. Brancher is based on the deep learning framework PyTorch. 

## Building probabilistic models ##
Probabilistic models are defined symbolically. Random variables can be created as follows:
```python
a = NormalVariable(loc = 0., scale = 1., name = 'a')
b = NormalVariable(loc = 0., scale = 1., name = 'b')
```
It is possible to chain together random variables by using arithmetic and mathematical functions:
```python
c = NormalVariable(loc = a**2 + BF.sin(b), 
                   scale = BF.exp(b), 
                   name = 'a')
```
In this way, it is possible to create arbitrarely complex probabilistic models. It is also possible to use all the deep learning tools of PyTorch in order to define probabilistic models with deep neural networks.

## Example: Autoregressive modeling ##

### Probabilistic model ###
Probabilistic models are defined symbolically:

```python
T = 20
driving_noise = 1.
measure_noise = 0.3
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'x0')
b = LogitNormalVariable(0.5, 1., 'b')

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
for t in range(1,T):
    x_names.append("x{}".format(t))
    y_names.append("y{}".format(t))
    x.append(NormalVariable(b*x[t-1], driving_noise, x_names[t]))
    y.append(NormalVariable(x[t], measure_noise, y_names[t]))
AR_model = ProbabilisticModel(x + y)
```


### Observe data ###
Once the probabilistic model is define, we can decide which variable is observed:

```python
[yt.observe(data[yt][:, 0, :]) for yt in y]
```

### Autoregressive variational distribution ###
The variational distribution can have an arbitrary structure:

```python
Qb = LogitNormalVariable(0.5, 0.5, "b", learnable=True)
logit_b_post = DeterministicVariable(0., 'logit_b_post', learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [DeterministicVariable(0., 'x0_mean', learnable=True)]
for t in range(1, T):
    Qx_mean.append(DeterministicVariable(0., x_names[t] + "_mean", learnable=True))
    Qx.append(NormalVariable(BF.sigmoid(logit_b_post)*Qx[t-1] + Qx_mean[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)
model.set_posterior_model(variational_posterior)
```

### Inference ###
Now that the models are spicified we can perform approximate inference using stochastic gradient descent:

```python
inference.perform_inference(AR_model, 
                            number_iterations=500,
                            number_samples=300,
                            optimizer="SGD",
                            lr=0.001)
```

