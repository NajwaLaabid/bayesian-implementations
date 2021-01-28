import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import seaborn
from matplotlib import pyplot as plt, mlab as mlab 

# generate 1-d data from a gaussian mixture model
def gmm(nb_clusters, hyperparams, batch_size):
    alpha = np.full(shape=nb_clusters, fill_value=hyperparams['alpha'], dtype='float32')
    
    theta = tfd.Dirichlet(concentration=alpha).sample() # assignments probability
    
    mu = tfd.Normal(loc=hyperparams['mu0'], scale=hyperparams['sigma0']).sample(nb_clusters) # centroids
    
    z = tfd.OneHotCategorical(probs=theta).sample(batch_size) # assignment indicators
    assignments = tf.argmax(z.numpy(), axis=1)
    
    means = tf.gather(mu, assignments) # mapping 
    stds = tf.fill(dims=batch_size, value=hyperparams['tau2'])
    x = tfd.Normal(loc=means, scale=stds).sample()

    return mu.numpy(), theta.numpy(), assignments.numpy(), x.numpy()

# implement the complete conditional for RV z given equation (127) in section 5.3
def sample_z(x, mu, hyperparams):
    # get proba dependent on likelihood of x in clusters
    tau2 = np.full(shape=mu.shape, fill_value=hyperparams['tau2'], dtype='float32')
    probability = tfd.Normal(loc=mu, scale=tau2).prob(x).numpy()
    if sum(probability) > 0: probability = probability/sum(probability)
    z = tfd.OneHotCategorical(probs=probability).sample().numpy()
    
    return z, probability
    
# implement the complete conditional for RV mu: equations (132) and (133) in section 5.3
def sample_mu(z, m, X, hyperparams):
    Nm = z.sum(axis=0)[m] # all occurences of centroid 
    if X.size > 0 : xm = np.mean(X, axis=0) # handle case where cluster empty
    else: xm = 0
    
    lambda2_m = 1/(Nm/hyperparams['tau2'] + 1/hyperparams['sigma0'])
    mu_m = lambda2_m*(xm*Nm/hyperparams['tau2'] + hyperparams['mu0']/hyperparams['sigma0'])
 
    mu = tfd.Normal(loc=mu_m, scale=lambda2_m).sample().numpy()

    return mu, mu_m, lambda2_m

def gibbs_sampler(iterations, nb_clusters, X, hyperparams):
    mu = np.full(shape=nb_clusters, fill_value=hyperparams['mu0'], dtype='float32') # init with mean of means
    z = np.full(shape=(X.shape[0], nb_clusters) , fill_value=0) # all assigned to first cluster

    trace = []
    for it in range(iterations):
        for i, xi in enumerate(X):
            z[i, :] = sample_z(xi, mu, hyperparams)[0]
        for m, mu_m in enumerate(mu):
            Xk = X[np.argmax(z, axis=-1) == m]
            mu[m] = sample_mu(z, m, Xk, hyperparams)[0]
        trace.append(np.array((z.copy(), mu.copy())))
    
    return trace[-1][1], trace[-1][0], trace