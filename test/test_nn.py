# TODO: import dependencies and write unit tests below
import pytest
import sklearn
from nn import nn
import numpy as np

def test_single_forward():
    #defining testing parameters (weights, biases, prev activation)
    input_dim, output_dim = 3, 2 
    W_test = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  
    b_test = np.array([[0.1], [0.2]])  
    A_prev_test = np.array([[0.7, 0.8, 0.9]]).T  
    activation = 'relu'
    #expected outputs
    Z_expected = np.dot(W_test, A_prev_test) + b_test
    A_expected = np.maximum(0, Z_expected)
    arch = [{'input_dim': input_dim, 'output_dim': output_dim, 'activation': activation}]
    test = nn.NeuralNetwork(nn_arch=arch,lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='mean_squared_error')
    #replace initial params with our testing params
    test._param_dict['W1'] = W_test
    test._param_dict['b1'] = b_test
    A_curr, Z_curr = test._single_forward(W_test, b_test, A_prev_test, activation)
    assert np.allclose(Z_curr, Z_expected), "Z_curr does not match expected values"
    assert np.allclose(A_curr, A_expected), "A_curr does not match expected values"
    print("test_single_forward passed successfully.")

def test_forward():
    nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    test = nn.NeuralNetwork(nn_arch, lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='mean_squared_error')
    #define testing parameters
    test._param_dict['W1'] = np.array([[0.1, 0.2], [0.3, 0.4]]) 
    test._param_dict['b1'] = np.array([[0.1], [0.2]])  
    test._param_dict['W2'] = np.array([[0.5, 0.6]])  
    test._param_dict['b2'] = np.array([[0.3]])    
    x_test = np.array([[0.1, 0.2]]).T  
    #manually do a forward pass
    Z1 = np.dot(test._param_dict['W1'], x_test) + test._param_dict['b1']
    A1 = np.maximum(0, Z1)  #relu
    Z2 = np.dot(test._param_dict['W2'], A1) + test._param_dict['b2']
    A2 = 1 / (1 + np.exp(-Z2))  #sigmoid
    #forward pass using my nn func
    output, _ = test.forward(x_test.T)  
    assert np.allclose(output, A2.T), "Output of forward method does not match expected values"
    print("test_forward passed successfully")
    
def test_single_backprop():
    nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}]
    test = nn.NeuralNetwork(nn_arch, lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='')

    #define paramaters
    W_curr = np.array([[0.1, 0.2], [0.3, 0.4]])
    b_curr = np.array([[0.1], [0.2]])
    A_prev = np.array([[0.5, 0.6]]).T #shape = [input,1]
    dA_curr = np.array([[0.25], [0.35]])  
    test._param_dict['W1'] = W_curr
    test._param_dict['b1'] = b_curr
    activation = 'relu'
    #manually calculating gradient:
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    dZ_curr = dA_curr * (Z_curr > 0).astype(float)  
    dW_expected = np.dot(dZ_curr, A_prev.T)
    db_expected = np.sum(dZ_curr, axis=1, keepdims=True)
    dA_prev_expected = np.dot(W_curr.T, dZ_curr)
    #using my nn single backprop function
    dW_calc, db_calc, dA_prev_calc = test._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

    assert np.allclose(dW_expected, dW_calc), "dW gradient not matching expected values."
    assert np.allclose(db_expected, db_calc), "db gradient not matching expected values."
    assert np.allclose(dA_prev_expected, dA_prev_calc), "dA_prev gradient not matching expected values."
    print("test_single_backprop passed successfully.")

def test_predict():
    nn_arch = [
        {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
        {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    test_nn = nn.NeuralNetwork(nn_arch, lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='mse')

    #testing parameters
    test_nn._param_dict['W1'] = np.array([[0.1, 0.2], [0.3, 0.4]])
    test_nn._param_dict['b1'] = np.array([[0.1], [0.2]])
    test_nn._param_dict['W2'] = np.array([[0.5, 0.6]])
    test_nn._param_dict['b2'] = np.array([[0.3]])
    x_test = np.array([[0.1, 0.2]])  
    #manual calculation
    Z1 = np.dot(test_nn._param_dict['W1'], x_test.T) + test_nn._param_dict['b1']
    A1 = np.maximum(0, Z1)  #relu
    Z2 = np.dot(test_nn._param_dict['W2'], A1) + test_nn._param_dict['b2']
    A2_expected = 1 / (1 + np.exp(-Z2))  #sigmoid

    #neural network prediction
    y_pred = test_nn.predict(x_test)
    assert np.allclose(y_pred, A2_expected.T), "Predicted values do not match expected values."
    print("test_predict passed successfully.")

def test_binary_cross_entropy():
    nn_arch = [{'input_dim': 1, 'output_dim': 1, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(nn_arch, lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    #parameters
    y_hat = np.array([[0.9, 0.2, 0.1]])  
    y = np.array([[1, 0, 1]])  
    #manual calculation of loss
    eps = 1e-12 
    y_hat_new = np.clip(y_hat, eps, 1 - eps)  
    expected_loss = -1/1 * np.sum(y * np.log(y_hat_new) + (1 - y) * np.log(1 - y_hat_new)) #here batch size = 1
    #using nn binary cross-entropy func to calculate loss
    calculated_loss = test_nn._binary_cross_entropy(y, y_hat)
    assert np.isclose(calculated_loss, expected_loss), "Binary cross entropy loss does not match expected value."
    print("test_binary_cross_entropy passed successfully.")

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass


test_single_forward()
test_forward()
test_single_backprop()
test_predict()
test_binary_cross_entropy()