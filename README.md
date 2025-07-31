# Deep-Learning-Concepts
We gonna learn about some important deep learning concepts, one must know.....

## What is Deep Learning?
- **Deep learning** is a subset of machine learning that uses artificial neural networks with multiple layers to analyze data and learn patterns.
- It’s inspired by how the human brain works.
- It uses structures called artificial neural networks that are made up of layers of interconnected nodes **(neurons)**.“Artificial” neural networks are inspired by the organic brain, translated to the computer.

<img width="926" height="384" alt="image" src="https://github.com/user-attachments/assets/207acff1-9d73-4414-9f01-8e057bd0be2e" />

- Here in this Example image, we represent single neuron. and sigmoid as activation function.
- As input you provide **Age** and output you will get if greater than 0.5 Person will buy insurance, or otherwise.

<img width="926" height="384" alt="image" src="https://github.com/user-attachments/assets/ae112851-61de-430b-badd-9db6301aed0f" />

##  Why “Deep” in Deep Learning?
The "**deep**" in Deep Learning comes from the many layers in the neural network:
1. **Input Layer**: Takes in the raw data (e.g., pixels in an image).
2. **Hidden Layers**: Multiple layers where learning happens by adjusting weights.
3. **Output Layer**: Produces the final prediction/classification.

## Activation Function
- Activation function in a neural network is a mathematical function which indicates if a perticular neuron is activated or not.
- It introduces non-linearity into the network, allowing it to learn complex patterns in data.
- To learn complex patterns, neural networks need non-linear functions, and that's exactly what activation functions provide. in other words activation function helps the network learn complex patterns, such as non-linear relationships between features and outputs.

### Types of Activation Functions 
1. Linear Activation Function
2. Sigmoid Function (Non linear)
3. Tanh Activation Function (Non Linear)
4. ReLU (Rectified Linear Unit)

### 1. Linear Activation Function
- A Linear Activation Function is the simplest type of activation used in neural networks. It does not modify the input—it just passes it through as output.
- Outputs the input value directly, Useful in the output layer for regression tasks where the output can take any real value.
- Using a linear activation in hidden layers makes the whole neural network equivalent to just one linear transformation.

$$
f(x) = x
$$

- **Diagram**
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/19f4c1ed-3b3a-4f24-93a6-2e94f58ff81d" />

### 2. Sigmoid Function
- The sigmoid activation function is a mathematical function which maps any input value to a range between 0 and 1, producing an "S" shaped curve.
- Commonly used in the output layer for binary classification tasks.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- **Diagram**
<img width="460" height="260" alt="image" src="https://github.com/user-attachments/assets/14d20f40-3922-40fe-9854-72fe6cd829a3" />

### 3. Tanh Activation Function
- The Tanh activation function is a non-linear function that transforms inputs into a range between -1 and 1, unlike sigmoid function which maps between range 0 to 1.
- Commonly used in the hidden layers.
- Preferred over sigmoid because outputs are zero-centered.

$$
f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

- **Diagram**
<img width="488" height="250" alt="image" src="https://github.com/user-attachments/assets/dcfb261d-09e6-4763-8923-373b8be5a11c" />


#### Zero-Centered
The function's output is symmetrical around zero, meaning negative inputs are mapped to negative outputs and positive inputs to positive outputs.
- Positive input ⇒ Positive output
- Negative input ⇒ Negative output
- Input = 0 ⇒ Output ≈ 0
This behavior makes the outputs symmetrical around zero, helping the neural network balance updates during training.

### 4.1 ReLu Activation Function
- It introduces non-linearity, allowing the network to learn complex patterns, and is defined as f(x) = max(0, x).
- This means, if the input value is positive it will take the positive value or if input is negative output will be 0.

$$
\text{ReLU}(x) = \max(0, x)
$$

                                                    This means:
                                                    - If x>0, output is 𝑥
                                                    - If x≤0, output is 0

- Relu is good for hidden layers, but as it produces output 0 for every negative value, which causes a problem called **dying ReLu**.

- **Diagram**
<img width="488" height="268" alt="image" src="https://github.com/user-attachments/assets/80965a58-e942-478d-a9ee-aec409118e5e" />

**Dying ReLU Problem**: 
If many neurons receive negative input during training, they may get stuck outputting 0 permanently.This means the neuron is not contributing to the network's learning process, and the gradient is zero during backpropagation,preventing the neuron's weights from being updated.

### 4.2 Leaky ReLU
- Leaky ReLU is a modified version of ReLU designed to fix the "dying ReLU" problem.
- Unlike ReLU, which outputs zero for all negative inputs, Leaky ReLU allows a small, non-zero gradient when the input is negative. which helps neurons to be active during training.

$$
\text{Leaky ReLU}(x) =
\begin{cases}
x, & \text{if } x > 0 \\
0.01 \cdot x, & \text{if } x \leq 0
\end{cases}
$$

<img width="492" height="314" alt="image" src="https://github.com/user-attachments/assets/2c0229e9-9659-4537-b1c0-20499e55612b" />

## Batch Gradient Descent and Stochastic Gradient Descent: 
- **Batch Gradient Descent (BGD)** and **Stochastic Gradient Descent (SGD)** are key optimization technique for reducing the **cost function** in machine learning, commonly applied in training models such as linear regression, logistic regression, and neural networks.
- These variants differ mainly in how they process data and optimize the model parameters.

### Batch Gradient Descent
- It computes the **gradient of the cost function** with respect to the model parameters using the **entire training dataset** in each iteration.
- While this method provides an accurate gradient calculation, it can become computationally intensive with very large datasets.

### Step-by-Step Process of Batch Gradient Descent
- **1.Initialize Weights and Biases:**
Assign initial values to the model’s parameters (weights and biases). These are typically initialized randomly or set to small values.

- **2.Compute Predictions:**
Use the current weights and biases to compute predictions for the entire training dataset. For each data point (xi, yi) in the dataset (where xi is the input and yi is the true output), calculate the predicted output yi.

- **3.Calculate the Cost & Gradients**
Compute the cost function (also loss function) using the predictions and true outputs across the entire dataset.

- **4.Update Weights and Biases**
Adjust the parameters in the direction that reduces the cost function, using the gradients and a learning rate.

$$w = w - \eta \cdot \frac{\partial J}{\partial w}$$
$$b = b - \eta \cdot \frac{\partial J}{\partial b}$$

- **5.Repeat Until Convergence**
Continue iterating through all above mentioned Steps for a set number of iterations (epochs) or until the cost function stops decreasing significantly.

<p align="center">
<img width="416" height="374" alt="image" src="https://github.com/user-attachments/assets/12cb6820-829f-4939-94d5-e79a44007d50" />
</p>

### Stochastic Gradient Descent
- Unlike Batch Gradient Descent (BGD), which processes the entire dataset in each iteration, SGD updates the model parameters using one randomly selected data point at a time.
- This makes the algorithm much faster since only a small fraction of the data is processed at each step.
- SGD is particularly useful when dealing with large datasets, where processing the entire dataset at once is computationally expensive.

### Step-by-Step Process of Stochastic Gradient Descent
-**1. Initialize Weights and Biases** :
same as batch.

-**2. Select a Random Data Point and Compute Predicted Value** :
Select a Random Data Point (xi,yi) from the training dataset. And Use the current weights and biases to compute the prediction.

-**3. Compute Gradients for the Selected Point**:
Calculate the gradient of the loss with respect to each parameter (weights and biases) using only the selected data point.

-**4. Update Weights and Biases**:
Adjust the parameters using the gradients and a learning rate.

-**5. Repeat Until Convergence**:
Repeat all Steps 1–4 for each data point in the dataset, typically in a random order.

<p align="center">
<img width="496" height="350" alt="image" src="https://github.com/user-attachments/assets/16f1c33e-7daa-4f17-a005-b502e3027116" />
</p>

### Dropout Regularization:
- Dropout is a powerful regularization technique widely used in deep learning to prevent overfitting in neural networks.
- Overfitting happens when a model captures not only the underlying patterns in the training data but also the noise and random fluctuations, resulting in poor generalization to new, unseen data.
- To prevent that, we drop some of the neurons in the hidden layers, so that model do not fit too well and cause overfitting problem.
- Dropout helps address this by introducing randomness during the training process.
- We can add dropout rate like 0.5 for 50% neuron dropout, we can set any percentage 0.3 etc.

**Diagram**

<img width="1062" height="640" alt="image" src="https://github.com/user-attachments/assets/2c8ac064-9bcd-4715-8e09-c98fddbf9282" />

**Notebook** : [drop_out_regularization.ipynb](drop_out_regularization.ipynb)
