from flax import linen as nn
import jax.numpy as jnp

############################################################
############################################################

class FlaxModel(nn.Module):
    '''
    A Flax model class that represents a neural network.

    Attributes:
        width (int): The number of neurons in each layer.
        depth (int): The number of layers in the neural network.
        activation_fn (object): The activation function to use in each layer.
    '''

    width: int
    depth: int
    activation_fn: object

    @nn.compact
    def __call__(self, X):
        '''
        Defines the computation performed at every call.

        Args:
            X (jnp.array): The input data.

        Returns:
            X (jnp.array): The output of the neural network.
        '''
        for _ in jnp.arange(self.depth):
            X = nn.Dense(self.width)(X)
            X = self.activation_fn(X)
        X = nn.Dense(1)(X)
        return X
    
############################################################
############################################################
