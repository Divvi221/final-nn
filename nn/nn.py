# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(W_curr,A_prev) + b_curr #linear transformation step

        if activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation == "relu":
            A_curr = self._relu(Z_curr)
        else:
            raise Exception("Wrong activation function: {}".format(activation))
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        A_curr = X.T #initially had .T
        #num_layers = len(self.arch) - 1
        for ind, layer in enumerate(self.arch,1): 
            #getting matrices from param dict and setting A_prev
            A_prev = A_curr #setting inputs as A_curr for first loop
            W_curr = self._param_dict['W' + str(ind)]
            b_curr = self._param_dict['b' + str(ind)]
            activation = layer["activation"]
            #using the single_forward function to do a forward pass for each iteration
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr,activation)
            #storing A_prev and Z_curr in 
            cache['A' + str(ind-1)] = A_prev #used to be ind - 1 ; and = A_prev
            cache['Z' + str(ind)] = Z_curr

        output = A_curr.T #this .T was not here originally
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #calculating dZ_curr
        if activation_curr == "sigmoid":
            dZ_curr = self._sigmoid_backprop(dA_curr,Z_curr) 
        elif activation_curr == "relu":
            dZ_curr = self._relu_backprop(dA_curr,Z_curr)

        m = A_prev.shape[1]
    
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dW_curr,db_curr,dA_prev

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}

        #calculate loss and dA_prev depending on activation func
        if self._loss_func == "binary_cross_entropy":
            dA_prev = self._binary_cross_entropy_backprop(y,y_hat)
        elif self._loss_func == "mean_squared_error":
            dA_prev = self._mean_squared_error_backprop(y,y_hat)
        else:
            raise Exception("Incorrect loss function")
        
        #iterate through layers backwards
        for idx_prev, layer in reversed(list(enumerate(self.arch, 1))):
            idx = idx_prev - 1
            activation_curr = layer["activation"]
            dA_curr = dA_prev

            #extract the matrices for the given layer
            A_prev = cache['A' + str(idx)]
            Z_curr = cache['Z' + str(idx_prev)]
            W_curr = self._param_dict['W' + str(idx_prev)]
            b_curr = self._param_dict['b' + str(idx_prev)]

            #use single backprop to do backprop for this layer
            dW_curr, db_curr, dA_prev = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            #store the gradients in grad_dict
            grad_dict["dW" + str(idx_prev)] = dW_curr
            grad_dict["db" + str(idx_prev)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        #for key in self._param_dict.keys():
        #    if key in grad_dict:
        #        #self._param_dict[key] -= self._lr * grad_dict[key]
        #        update_amount = self._lr * grad_dict[key]
        #        print(f"Updating {key}: max update amount {np.max(np.abs(update_amount))}")
        #        self._param_dict[key] -= update_amount
        for idx, layer in enumerate(self.arch, 1):
            #dictionary keys to access the weights, biases and their gradients
            W = f'W{idx}'
            dW = f'dW{idx}'

            b = f'b{idx}'
            db = f'db{idx}'

            if dW in grad_dict and db in grad_dict:
                self._param_dict[W] = self._param_dict[W] - (self._lr * grad_dict[dW])
                self._param_dict[b] = self._param_dict[b] - (self._lr * grad_dict[db])

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            #forward pass
            output_train,cache_t = self.forward(X_train)
            output_val, cache_val = self.forward(X_val)

            #calculate loss
            if self._loss_func == "binary_cross_entropy":
                training_loss = self._binary_cross_entropy(y_train,output_train)
            elif self._loss_func == "mean_squared_error":
                training_loss = self._mean_squared_error(y_train,output_train)
            #backpropagation
            grad_dict_t = self.backprop(y_train, output_train, cache_t)
            #update params
            self._update_params(grad_dict_t)

            #store training loss per epoch
            per_epoch_loss_train.append(training_loss)

            if self._loss_func == 'binary_cross_entropy':
                val_loss = self._binary_cross_entropy(y_val, output_val)
            elif self._loss_func == "mean_squared_error":
                val_loss = self._mean_squared_error(y_val,output_val)

            #store validation loss per epoch
            per_epoch_loss_val.append(val_loss)
    
        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sigmoid = self._sigmoid(Z)
        print("sig shape:",sigmoid.T.shape)
        print("dA:",dA.shape) #dA shape is the problem
        dZ = dA.T * sigmoid * (1-sigmoid) #using chain rule and derivative of sigmoid this .T was not here before
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        relu = np.maximum(0,Z)
        return relu

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True) #this .T was not here before
        dZ[Z <= 0] = 0 #using array masking to make all the negative and zero values = 0. The positive values can stay positive here (but technically in the relu derivative they should = 1).
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        eps = 1e-14 #epsilon
        y_hat_new = np.clip(y_hat, eps, 1 - eps) #prevent log(0)
        loss = (-1/self._batch_size) * np.sum(y * np.log(y_hat_new) + (1 - y) * np.log(1 - y_hat_new)) #y_hat here used to be y_hat_new
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        eps = 1e-14
        y_hat_new = np.clip(y_hat, eps, 1 - eps)
        #gradient calculation
        #N = y.shape[0]
        #dA = - (np.divide(y,y_hat_new) - np.divide((1-y),(1-y_hat_new))) # * (1/N)
        dA = -1/ self._batch_size * np.divide(y - y_hat_new, y_hat_new * (1 - y_hat_new))
        return dA


    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        loss = np.sum((y - y_hat) ** 2)/ self._batch_size
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        dA = (2.0/self._batch_size) * (y_hat - y) #derivative
        return dA
