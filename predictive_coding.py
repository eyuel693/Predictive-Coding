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

W1, W2, W3, L2, L3 = init_predictive_coding_params()
W_list = [W1, W2, W3]
L_list = [L2, L3]

print("Initial weights and lateral matrices:")
print("W1 shape:", W1.shape, "\nW1:\n", W1[:3, :3])
print("W2 shape:", W2.shape, "\nW2:\n", W2[:3, :3])
print("W3 shape:", W3.shape, "\nW3:\n", W3[:3, :3])
print("L2 shape:", L2.shape, "\nL2:\n", L2[:3, :3])
print("L3 shape:", L3.shape, "\nL3:\n", L3[:3, :3])

num_training_epochs = 5

print("\n--- Starting Training of Predictive Coding Model ---")
for epoch in range(num_training_epochs):
    print(f"\nEpoch {epoch + 1}/{num_training_epochs}")
    for i in range(m_train):
        current_sample_input = X_train[:, i]
        current_sample_label = Y_train[i]

        r1 = current_sample_input.copy()
        r2 = np.random.rand(hidden_size_1) * 0.1
        r3 = np.random.rand(hidden_size_2) * 0.1
        r4 = np.random.rand(output_size) * 0.1
        r_list = [r1, r2, r3, r4]

        for inf_step in range(num_inference_iterations):
            r_list, e_list = update_representations(r_list, W_list, L_list, learning_rate_inference, lateral_strength, label=current_sample_label)
            total_inf_error = np.sum([np.linalg.norm(e) ** 2 for e in e_list])
            if total_inf_error < convergence_threshold:
                break

        W_list, dW_norms, dW_list = update_weights(r_list, e_list, W_list, gamma, dt)

        if (i + 1) % 5000 == 0:
            e1, e2, e3, e4 = e_list
            dW1, dW2, dW3 = dW_list
            predictions = get_predictions(r_list, W_list)
            predicted_class = np.argmax(predictions)
            print(f"  Processed {i + 1}/{m_train} samples. Current sample label: {current_sample_label}, Predicted: {predicted_class}")
            print(f"    Errors: e1: {np.linalg.norm(e1):.6f}, e2: {np.linalg.norm(e2):.6f}, e3: {np.linalg.norm(e3):.6f}, e4: {np.linalg.norm(e4):.6f}")
            print(f"    Weight updates: dW1: {np.mean(np.abs(dW1)):.6f}, dW2: {np.mean(np.abs(dW2)):.6f}, dW3: {np.mean(np.abs(dW3)):.6f}")

    correct_predictions_dev = 0
    total_dev_samples = X_dev.shape[1]
    print(f"\n  Evaluating on Dev Set after Epoch {epoch + 1}...")
    for i in range(total_dev_samples):
        dev_input_sample = X_dev[:, i]
        dev_label = Y_dev[i]

        r1_dev = dev_input_sample.copy()
        r2_dev = np.random.rand(hidden_size_1) * 0.1
        r3_dev = np.random.rand(hidden_size_2) * 0.1
        r4_dev = np.random.rand(output_size) * 0.1
        r_list_dev = [r1_dev, r2_dev, r3_dev, r4_dev]

        for inf_step_dev in range(num_inference_iterations):
            r_list_dev, e_list_dev = update_representations(r_list_dev, W_list, L_list, learning_rate_inference, lateral_strength, label=dev_label)
            total_inf_error_dev = np.sum([np.linalg.norm(e) ** 2 for e in e_list_dev])
            if total_inf_error_dev < convergence_threshold:
                break

        final_output_probs = get_predictions(r_list_dev, W_list)
        predicted_class_dev = np.argmax(final_output_probs)
        if predicted_class_dev == dev_label:
            correct_predictions_dev += 1

        if i == 0:
            e1_dev, e2_dev, e3_dev, e4_dev = e_list_dev
            print(f"    Dev sample 1 errors: e1: {np.linalg.norm(e1_dev):.6f}, e2: {np.linalg.norm(e2_dev):.6f}, e3: {np.linalg.norm(e3_dev):.6f}, e4: {np.linalg.norm(e4_dev):.6f}")

    accuracy_dev = correct_predictions_dev / total_dev_samples
    print(f"  Dev Set Accuracy after Epoch {epoch + 1}: {accuracy_dev * 100:.2f}%")

print("\n--- Training Complete ---")

print("\n--- Testing a single example after training ---")
test_index = 0
test_input = X_train[:, test_index]
true_label = Y_train[test_index]

r1_test = test_input.copy()
r2_test = np.random.rand(hidden_size_1) * 0.1
r3_test = np.random.rand(hidden_size_2) * 0.1
r4_test = np.random.rand(output_size) * 0.1
r_list_test = [r1_test, r2_test, r3_test, r4_test]

print(f"Running inference for test sample (true label: {true_label})...")
for inf_step_test in range(num_inference_iterations):
    r_list_test, e_list_test = update_representations(r_list_test, W_list, L_list, learning_rate_inference, lateral_strength, label=true_label)
    total_inf_error_test = np.sum([np.linalg.norm(e) ** 2 for e in e_list_test])
    if total_inf_error_test < convergence_threshold:
        print(f"    Inference converged at step {inf_step_test + 1}")
        break

e1_test, e2_test, e3_test, e4_test = e_list_test
print(f"    Test sample errors: e1: {np.linalg.norm(e1_test):.6f}, e2: {np.linalg.norm(e2_test):.6f}, e3: {np.linalg.norm(e3_test):.6f}, e4: {np.linalg.norm(e4_test):.6f}")

final_output_probs_test = get_predictions(r_list_test, W_list)
predicted_class_test = np.argmax(final_output_probs_test)

print(f"Predicted class: {predicted_class_test}")
print(f"True label: {true_label}")
