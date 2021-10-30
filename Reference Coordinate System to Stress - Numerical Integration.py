#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


RCS_original= np.array([[-1, -0.987063428, -0.974129442, -0.961198042, -0.948269226, -0.935342994, -0.922419344, -0.909498275, -0.896579787, -0.870750547, -0.735315981, -0.664487066, -0.600164371, -0.51664037, -0.394760493, -0.330705304, -0.260317778, -0.196395511,-0.126153894, -0.062363997, 0.007732313, 0.071390387, 0.141341991, 0.204868786, 0.27467628, 0.338072335, 0.407736312, 0.471002165, 0.540523214, 0.603659398, 0.673038104, 0.73604515, 0.805282094, 0.817862672, 0.937256291, 1.000006639]])
t_strain_original=np.array([[0, 0.019998, 0.039992, 0.059982, 0.079968, 0.09995, 0.1199281, 0.1399021, 0.1598721, 0.1998003, 0.4091618, 0.5186527, 0.6180859, 0.7472015, 0.9356095, 1.0346292, 1.1434378, 1.242252, 1.350835, 1.4494446, 1.557803, 1.6562088, 1.7643435, 1.8625464, 1.9704583, 2.0684591, 2.1761492, 2.2739487, 2.3814178, 2.4790169, 2.586266,2.6836654, 2.7906953, 2.810143, 2.9947076, 3.0917103]])
t_stress_original= np.array([[0, 0.340068, 0.710284, 1.10066, 1.5012, 1.96196, 2.332796, 2.65371, 3.114976, 3.3066, 3.805539, 4.0208, 4.22604, 4.373075, 4.532206, 4.607424, 4.683245, 4.7385, 4.794328, 4.849788, 4.885517, 4.900661, 4.916864, 4.921744, 4.926117, 4.930947, 4.93626, 4.94109, 4.946403, 4.951233, 4.956546, 4.960832, 4.965274, 4.9666, 4.983792, 4.996952]])

RCS = RCS_original.T
t_strain=t_strain_original.T
t_stress = t_stress_original.T

plt.plot(t_stress , RCS)
plt.show()


# In[3]:


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)        
        self.dinputs = np.dot(dvalues, self.weights.T)
      
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  
      
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()       
        
class Optimizer_Adam:
    def __init__(self, learning_rate=0.02, decay=0.0001, epsilon=1e-7, beta_1=0.9, beta_2=0.99):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights        
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases  
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))       
        bias_momentums_corrected = layer.bias_momentums /(1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2        
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache /(1 - self.beta_2 ** (self.iterations + 1))        
        bias_cache_corrected = layer.bias_cache /(1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate *weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)   
        layer.biases += -self.current_learning_rate * bias_momentums_corrected /(np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1
        
class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_MeanSquaredError(Loss): 
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


# In[4]:


dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam()

accuracy_precision = np.std(t_stress) / 250

for epoch in range(500000):
    dense1.forward(RCS)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_function.calculate(activation3.output, t_stress)
    
    loss = data_loss 
    
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - t_stress) <accuracy_precision)
    
    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy}, ' +
              f'loss: {loss} (' +
              f'data_loss: {data_loss}, ' +
              f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_function.backward(activation3.output, t_stress)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
    


# In[6]:


import matplotlib.pyplot as plt
X_test = np.array([[-1, -0.987063428, -0.974129442, -0.961198042, -0.948269226, -0.935342994, -0.922419344, -0.909498275, -0.896579787, -0.870750547, -0.735315981, -0.664487066, -0.600164371, -0.51664037, -0.394760493, -0.330705304, -0.260317778, -0.196395511,-0.126153894, -0.062363997, 0.007732313, 0.071390387, 0.141341991, 0.204868786, 0.27467628, 0.338072335, 0.407736312, 0.471002165, 0.540523214, 0.603659398, 0.673038104, 0.73604515, 0.805282094, 0.817862672, 0.937256291, 1.000006639]])
X_test= X_test.T


dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)


plt.scatter(RCS, t_stress*100 , color='blue', label='Experimental data')
plt.plot(X_test, 100*activation3.output, color='red', label='Output from Neural Network')
plt.legend()

plt.xlabel("Reference Coordinate (r)")
plt.ylabel("Stress(MPa)")
plt.show()


# In[20]:


r1 = -0.96392371  ; r2 = -1

N = 10000

r_original = np.linspace(r1, r2, N+1)
r = np.reshape (r_original, (-1,1))
#print(r)

dense1.forward(r)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

t_stress= 100*activation3.output
t_strain= 0.015458499988425798*r+ 0.015458499989773924 

y = t_stress*t_strain

y_right = y[1:]
y_left = y[:-1]

dr= (r1- r2)/N
A = (dr/2)*np.sum(y_right+y_left)

print(A)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




