import numpy as np 

learn_rate = 0.4
n = 500000

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([
    [0,0,0,0,0,0,0,0,0], # Full White
    [0,1,0,1,1,1,0,1,0], # Plus
    [1,0,1,0,1,0,1,0,1], # X 
    [1,1,1,1,0,1,1,1,1],  # Circle / Square (No middle)
    [0,0,0,1,1,1,0,0,0] # Middle Full Only
    ])
actual_output = np.array([
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
    ])

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 9,9,5
weights = [np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons)),np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))]
bias = [np.random.uniform(size=(1,hiddenLayerNeurons)),np.random.uniform(size=(1,outputLayerNeurons))]

for _ in range(n):
    # Feedforward:
    hidden_layer_activation = np.dot(inputs,weights[0]) + bias[0]
    hidden_layer = sigmoid(hidden_layer_activation)
    output_activation = np.dot(hidden_layer,weights[1]) + bias[1]
    predicted_output = sigmoid(output_activation)

    #Backpropagation
    error = (actual_output - predicted_output)**2
    d_weights = []
    d_bias = []
    d_weights.append(2 * (actual_output - predicted_output) * sigmoid_derivative(predicted_output))
    d_bias.append(sigmoid_derivative(output_activation) * 2 * (predicted_output - actual_output))
    d_weights.append(np.dot(d_weights[0],weights[1].T) * sigmoid_derivative(hidden_layer))

    weights[1] += np.dot(hidden_layer.T,d_weights[0]) * learn_rate
    bias[1] += np.sum(d_weights[0],axis=0,keepdims=True) * learn_rate 
    weights[0] += np.dot(inputs.T,d_weights[1]) * learn_rate
    bias[0] += np.sum(d_weights[1],axis=0,keepdims=True) * learn_rate


inputs = np.array([
    [1,1,1,1,1,1,1,1,1], # Full Black
    [0,1,0,1,0,1,0,1,0], # Plus w/o Middle
    [1,1,1,0,0,0,1,1,1], # Top/Bot Full Color 
    [0,0,0,0,1,0,0,0,0]  # 1 Middle
    ])
hidden_layer_activation = np.dot(inputs,weights[0]) + bias[0]
hidden_layer = sigmoid(hidden_layer_activation)
output_activation = np.dot(hidden_layer,weights[1]) + bias[1]
predicted_output = sigmoid(output_activation)
print(predicted_output)
for a in predicted_output:
    for i in range(0,len(a)):
        if(a[i] == np.max(a)):
            a[i] = 1
        else:
            a[i] = 0
    print(a)