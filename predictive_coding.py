import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

data = pd.read_csv('data/train.csv')  
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)


data_dev = data[0:1000].T
Y_dev = data_dev[0]  
X_dev = data_dev[1:n]  
X_dev = X_dev / 255.  


data_train = data[1000:m].T
X_train = data_train[1:n] 
Y_train = data_train[0]  
X_train = X_train / 255.  
_, m_train = X_train.shape


input_size = 784 
hidden_size_1 = 128  
hidden_size_2 = 64  
output_size = 10  


lateral_strength = 0.1
learning_rate_inference = 0.1
learning_rate_weights = 0.01
gamma = 0.1
dt = 0.1
num_inference_iterations = 50
convergence_threshold = 1e-4

def init_predictive_coding_params():
    W1 = np.random.randn(hidden_size_1, input_size) * 0.01
    W2 = np.random.randn(hidden_size_2, hidden_size_1) * 0.01
    W3 = np.random.randn(output_size, hidden_size_2) * 0.01
    L2 = np.random.randn(hidden_size_1, hidden_size_1) * 0.01
    np.fill_diagonal(L2, 0)
    L3 = np.random.randn(hidden_size_2, hidden_size_2) * 0.01
    np.fill_diagonal(L3, 0)
    return W1, W2, W3, L2, L3

def relu(z):
    return np.maximum(z, 0)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def compute_error(actual_activity, predicted_activity):
    return actual_activity - predicted_activity

def compute_output_error(r4, label):
    one_hot = np.zeros(output_size)
    one_hot[int(label)] = 1.0
    return r4 - one_hot

def update_representations(r_list, W_list, L_list, lr_inf, lat_strength, label=None):
    r1, r2, r3, r4 = r_list
    W1, W2, W3 = W_list
    L2, L3 = L_list


    pred_r1_from_r2 = relu(np.dot(W1.T, r2))
    pred_r2_from_r3 = relu(np.dot(W2.T, r3))
    pred_r3_from_r4 = relu(np.dot(W3.T, r4))

    e1 = compute_error(r1, pred_r1_from_r2)
    e2 = compute_error(r2, pred_r2_from_r3)
    e3 = compute_error(r3, pred_r3_from_r4)
    e4 = compute_output_error(r4, label) if label is not None else np.zeros_like(r4)

    grad_r2 = np.dot(W1, e1) - e2 + lat_strength * np.dot(L2, r2)
    r2 += lr_inf * grad_r2
    r2 = sigmoid(r2)

    grad_r3 = np.dot(W2, e2) - e3 + lat_strength * np.dot(L3, r3)
    r3 += lr_inf * grad_r3
    r3 = sigmoid(r3)

    grad_r4 = np.dot(W3, e3) - e4
    r4 += lr_inf * grad_r4
    r4 = softmax(r4)

    return [r1, r2, r3, r4], [e1, e2, e3, e4]

def update_weights(r_list, e_list, W_list, gamma, dt):
    r1, r2, r3, r4 = r_list
    e1, e2, e3, e4 = e_list
    W1, W2, W3 = W_list
    dW_norms = []


    r1_act = sigmoid(r1)
    r2_act = sigmoid(r2)
    norm = np.linalg.norm(r2_act) * np.linalg.norm(r1_act) + 1e-6
    dW1 = gamma * np.outer(r2_act, r1_act) / norm  
    W1 += dW1 * dt
    W1 = np.clip(W1, -1.0, 1.0)
    dW_norms.append(np.mean(np.abs(dW1)))

    r2_act = sigmoid(r2)
    r3_act = sigmoid(r3)
    norm = np.linalg.norm(r3_act) * np.linalg.norm(r2_act) + 1e-6
    dW2 = gamma * np.outer(r3_act, r2_act) / norm  
    W2 += dW2 * dt
    W2 = np.clip(W2, -1.0, 1.0)
    dW_norms.append(np.mean(np.abs(dW2)))

    r3_act = sigmoid(r3)
    norm = np.linalg.norm(e4) * np.linalg.norm(r3_act) + 1e-6
    dW3 = gamma * np.outer(e4, r3_act) / norm 
    W3 += dW3 * dt
    W3 = np.clip(W3, -1.0, 1.0)
    dW_norms.append(np.mean(np.abs(dW3)))
    return [W1, W2, W3], dW_norms, [dW1, dW2, dW3]

def get_predictions(r_list, W_list):
    r1, r2, r3, r4 = r_list
    W1, W2, W3 = W_list
    h1 = relu(np.dot(W1, r1))
    h2 = relu(np.dot(W2, h1))
    out = softmax(np.dot(W3, h2))
    return out

def get_accuracy(predictions, Y_true):
    predicted_classes = np.argmax(predictions, axis=0)
    return np.sum(predicted_classes == Y_true) / Y_true.size
