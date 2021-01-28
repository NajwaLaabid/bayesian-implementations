import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def get_normal_params(arr):
	mean = np.mean(arr)
	variance = np.var(arr)
	sigma = np.sqrt(variance)

	x = np.linspace(min(arr), max(arr), 100)

	return x, mean, sigma

def plot_histograms(data, title=""):
	x, mean, sigma = get_normal_params(data)

	plt.figure()
	plt.hist(data, density=True)
	plt.plot(x, tfd.Normal(mean, sigma).prob(x))
	plt.title(title)
