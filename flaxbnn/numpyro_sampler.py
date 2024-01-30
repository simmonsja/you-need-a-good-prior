import pickle

# NumPyro for proabilistic programming
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.diagnostics import hpdi
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
from numpyro.contrib.module import random_flax_module, flax_module
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value, init_to_sample, init_to_feasible
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from flax import linen as nn

from .flax_bnn import FlaxModel

############################################################
############################################################

# numpyro model class
class NumPyroFlaxBNN():
    '''
    A class that represents a Bayesian Neural Network using NumPyro and Flax.

    Attributes:
        width (int): The number of neurons in each layer.
        depth (int): The number of layers in the neural network.
        activation_fn (object): The activation function to use in each layer.
        flax_module (object): The Flax module representing the neural network.
        model (object): The NumPyro model.
        samples (object): The samples from the posterior distribution.
    '''

    ############################################################
    ############################################################
 
    # init with width, depth and activation functin holders
    def __init__(self, width, depth, activation_fn=nn.tanh):
        '''
        Initializes the class with the given width, depth, and activation function.

        Args:
            width (int): The number of neurons in each layer.
            depth (int): The number of layers in the neural network.
            activation_fn (object): The activation function to use in each layer.
        '''
        self.width = width
        input_dim = 1
        output_dim = 1
        self.width_array = np.array([input_dim]+[width]*depth+[output_dim])
        self.depth = depth
        self.activation_fn = activation_fn
        self.flax_module = None
        self.model = None
        self.pretrain_model = None
        self.guide = None
        self.samples = None# optimise the 
    
    ############################################################
    ############################################################
    
    # define the flax model
    def define_flax_model(self):
        '''
        Defines the Flax model with the given width, depth, and activation function.
        '''
        self.flax_module = FlaxModel(self.width, self.depth, self.activation_fn)
    
    ############################################################
    ############################################################
        
    # define the numpyro model
    def define_numpyro_model(self,prior=None):
        '''
        Defines the NumPyro model with the given prior.

        Args:
            prior (dict): The prior distributions for the parameters.

        Notes on prior:
        The `prior` dictionary defines the prior distributions for the parameters of a Flax module in a NumPyro model. Each key corresponds to a parameter of the Flax module, with the format `'Dense_{layer_num}.{parameter}'`. The value is a tuple defining the mean and standard deviation of a Normal distribution, used as the prior for that parameter. Use the `convert_to_flax_prior` to convert a dictionary of priors for a NumPyro model to a dictionary of priors for a Flax module in a NumPyro model.
        '''
        def numpyro_model(X, Y=None):
            '''
            Defines a Bayesian Neural Network model using NumPyro and Flax.

            Args:
                X (jnp.array): The input data.
                Y (jnp.array): The output data. Default is None.
            '''
            # X_shape = X.shape[1]

            sigma_meas = numpyro.sample("sigma_meas", dist.Exponential(1))

            if not prior is None:
                flax_priors = {k: dist.Normal(v[0],v[1]) for k,v in prior.items()}
            else:
                flax_priors = {
                    'Dense_0.bias':dist.Normal(0,1),
                    'Dense_0.kernel':dist.Normal(0,1) #/self.width
                }

            nnet = random_flax_module(
                "bnn", 
                self.flax_module,
                input_shape=(X.shape[1],),
                # apply_rng=["dropout"],
                prior=flax_priors
            )
            y_out = nnet(X)

            # record our model prediction
            numpyro.deterministic('y_out',y_out)
            # and define the likelihood function
            numpyro.sample("obs", dist.Normal(y_out, sigma_meas), obs=Y)
        self.model = numpyro_model
    
    ############################################################
    ############################################################
        
    def define_guide(self):
        def numpyro_guide(X, Ygp=None):
            '''
            Guide description: Gaussian guide
            '''
            for ii in np.arange(self.depth+1):
                kernel_std = nn.activation.softplus(numpyro.param(
                    'Dense_{}.kernel_std'.format(ii),
                    0.1,
                    constraint=dist.constraints.real #softplus_positive
                ))
                bias_std = nn.activation.softplus(numpyro.param(
                    'Dense_{}.bias_std'.format(ii),
                    0.1,
                    constraint=dist.constraints.real #softplus_positive
                ))
                numpyro.sample('W{}'.format(ii), dist.Normal(0, kernel_std/jnp.sqrt(self.width_array[ii])).expand([self.width_array[ii], self.width_array[ii+1]]))
                numpyro.sample('b{}'.format(ii), dist.Normal(0, bias_std).expand([self.width_array[ii+1]]))

            sigma_scale = numpyro.param("sigma_scale", 1, constraint=dist.constraints.positive)
            numpyro.sample("sigma_meas", numpyro.distributions.HalfNormal(sigma_scale))
                
        self.guide = numpyro_guide

    ############################################################
    ############################################################

    # define the numpyro model
    def define_numpyro_pretrain_model(self):
        def numpyro_model(X, Ygp=None):
            '''
            This is an inelegant placeholder until I can sourt out a guide that talks to a flax module..
            '''

            sigma_meas = numpyro.sample("sigma_meas", dist.Exponential(1))  

            nnparams = {}
            # define the Ws and bs
            for ii in np.arange(self.depth+1):
                nnparams['W{}'.format(ii)] = numpyro.sample(
                    'W{}'.format(ii),
                    dist.Normal(0, 1).expand(
                        [self.width_array[ii], self.width_array[ii+1]]
                    )
                )
                nnparams['b{}'.format(ii)] = numpyro.sample(
                    'b{}'.format(ii),
                    dist.Normal(0, 1).expand(
                        [self.width_array[ii+1]]
                    )
                )
            # forward run
            hvals = X
            for ii in np.arange(self.depth):
                hvals = self.activation_fn(
                    jnp.matmul(
                        hvals,
                        nnparams['W{}'.format(ii)]
                    ) + nnparams['b{}'.format(ii)]
                )
            # final output
            y_out = jnp.matmul(hvals, nnparams['W{}'.format(self.depth)]) + nnparams['b{}'.format(self.depth)]

            if not Ygp is None:
                Y = Ygp.sample(numpyro.prng_key(), shape=(1,)).T
            else:
                Y = None
            # record our model prediction
            numpyro.deterministic('y_out',y_out)
            # and define the likelihood function
            numpyro.sample("obs", dist.Normal(y_out, sigma_meas), obs=Y)
        self.pretrain_model = numpyro_model

    ############################################################
    ############################################################
        
    def model_predict(self, rng_key, X, num_samples=1000, prior=True):
        '''
        Predicts the output for the given input data.

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            num_samples (int): The number of samples to draw from the posterior.
            prior (bool): Whether to use the prior or the posterior for prediction.

        Returns:
            jnp.array: The predicted output.
        '''

        rng, _ = jax.random.split(rng_key)
        if prior:
            samples = Predictive(
                self.model, num_samples=num_samples, return_sites=['y_out']
            )(
                rng, X=jnp.array(X),  Y=None
            )
        else:
            samples = Predictive(
                self.model, num_samples=num_samples, return_sites=['y_out'],
                posterior_samples=self.samples
            )(
                rng, X=jnp.array(X),  Y=None,
            ) 
        return samples['y_out'].squeeze().T
    
    ############################################################
    ############################################################
        
    def train_model(self,rng_key,X,Y,num_samples,num_warmup,num_chains,max_tree):
        '''
        Trains the model with the given input and output data.

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            Y (jnp.array): The output data.
            num_samples (int): The number of samples to draw from the posterior.
            num_warmup (int): The number of warmup steps before sampling.
            num_chains (int): The number of Markov chains to run in parallel.
            max_tree (int): The maximum tree depth for the NUTS sampler.
        '''
        rng, _ = jax.random.split(rng_key)
        nuts = NUTS(self.model)
        mcmc_obj = MCMC(
            nuts, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            num_chains=num_chains
        )
        mcmc_obj.run(rng, X = jnp.array(X), Y = jnp.array(Y))

        # get the samples which will form our posterior
        samples = mcmc_obj.get_samples()

        # # get the samples for predictive uncertainty (our linear model + error)
        # posterior_predictive = Predictive(
        #     self.model, posterior_samples=samples
        # )(rng, X = jnp.array(X),  Y = None)

        self.samples = samples

    ############################################################
    ############################################################
    
    def train_prior_model(self, rng_key, X, Y, num_samples, num_chains, step_size=0.01):
        '''
        Trains the prior model using Stochastic Variational Inference (SVI).

        Args:
            rng_key (object): The random number generator key.
            X (jnp.array): The input data.
            Y (jnp.array): The output data.
            num_samples (int): The number of samples to draw from the posterior.
            num_chains (int): The number of Markov chains to run in parallel.
            step_size (float): The step size for the RMSPropMomentum optimizer. Default is 0.01.
        '''
        elbo = Trace_ELBO(num_particles=num_chains)
        optimizer = numpyro.optim.RMSPropMomentum(step_size=step_size)
        #Clipped,clip_norm=10.0)

        svi = SVI(
            self.pretrain_model, self.guide, optimizer, loss=elbo, 
            X=jnp.array(X), Ygp=Y
        )
        svi_result = svi.run(rng_key, num_samples)
        self.params = svi_result.params
        self.losses = svi_result.losses

    ############################################################
    ############################################################    

############################################################
############################################################
# Helper functions
############################################################
############################################################

def convert_SVI_prior_to_flax(ckpt_path, width_array):

    '''
    Converts the state of a Stochastic Variational Inference (SVI) model saved in a checkpoint file to a Flax model prior.

    Args:
        ckpt_path (str): The path to the checkpoint file.
        width_array (list): A list of the number of neurons in each layer of the Flax model.

    Returns:
        flax_prior_def (dict): A dictionary where each key is a parameter of the Flax model and the value is a tuple defining the mean and standard deviation of a Normal distribution, used as the prior for that parameter.
    '''

    # pickle load state dict
    with open(ckpt_path, 'rb') as f:
        opt_prior_state = pickle.load(f)

    # convert torch state to flax state
    flax_prior_def = {}
    for ii, (k,v) in enumerate(opt_prior_state.items()):
        layer_num = ii // 2
        if 'W_std' in k:
            # softplus and scaled_variance to width
            flax_prior_def['Dense_{}.kernel'.format(layer_num)] = (
                0, nn.activation.softplus(v)/jnp.sqrt(width_array[layer_num])
            )
        elif 'b_std' in k:
            # softplus
            flax_prior_def['Dense_{}.bias'.format(layer_num)] = (0, nn.activation.softplus(v))

    return flax_prior_def

############################################################
############################################################

def simple_prior_to_flax(width_array):
    '''
    Creates a simple prior for a Flax model where the standard deviation for the 'kernel' parameters is 1/sqrt(number of neurons in the layer) and for the 'bias' parameters it is 1, both with mean 0.

    Args:
        width_array (list): A list of the number of neurons in each layer of the Flax model.

    Returns:
        std_prior_def (dict): A dictionary where each key is a parameter of the Flax model and the value is a tuple defining the mean and standard deviation of a Normal distribution, used as the prior for that parameter.
    '''

    # and get simple prior
    std_prior_def = {}
    for ii in jnp.arange(width_array.__len__()*2):
        layer_num = ii // 2
        std_prior_def['Dense_{}.kernel'.format(layer_num)] = (0, 1/jnp.sqrt(width_array[layer_num]))
        std_prior_def['Dense_{}.bias'.format(layer_num)] = (0, 1)
    return std_prior_def

############################################################
############################################################
