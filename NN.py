class DenseNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self._initialize_layers()
    
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def sigmoid_derivative(x):
        sigmoid_x = sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def _initialize_layers(self):
        self.input_shape = None
        for i in range(self.num_layers):
            if i == 0:
                num_neurons, activation, input_shape = self.layers[i]
            else:
                num_neurons, activation = self.layers[i]
            self.layers[i] = {
                'weights': np.random.normal(0, 1/num_neurons, (input_shape[0], num_neurons)),  #np.random.randn(input_shape[0], num_neurons) * 0.01,
                'biases': np.zeros((1, num_neurons)),
                'activation': activation,
                'input_shape': input_shape
            }
            self.input_shape = (num_neurons,)
    
    def forward(self, inputs):
        for layer in self.layers:
            weights = layer['weights']
            biases = layer['biases']
            activation = layer['activation']
            inputs = np.dot(inputs, weights) + biases
            if activation == 'relu':
                inputs = self.relu(inputs)
            elif activation == 'sigmoid':
                inputs = self.sigmoid(inputs)
        return inputs
    
    def backward(self, grad):
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            activation = layer['activation']
            inputs = self.layers[i - 1]['output'] if i > 0 else self.layers[i]['inputs']
            if activation == 'relu':
                grad *= self.relu_derivative(inputs)
            elif activation == 'sigmoid':
                grad *= self.sigmoid_derivative(-inputs)
            layer['grad_weights'] = np.dot(inputs.T, grad)
            layer['grad_biases'] = np.sum(grad, axis=0, keepdims=True)
            grad = np.dot(grad, layer['weights'].T)
        return grad
    
    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer['weights'] -= learning_rate * layer['grad_weights']
            layer['biases'] -= learning_rate * layer['grad_biases']
            
    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01, dropout_rate=None):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                # Forward pass
                inputs = x
                for layer in self.layers:
                    layer['inputs'] = inputs
                    weights = layer['weights']
                    biases = layer['biases']
                    activation = layer['activation']
                    inputs = np.dot(inputs, weights) + biases
                    if activation == 'relu':
                        inputs = self.relu(0, inputs)
                    elif activation == 'sigmoid':
                        inputs = self.sigmoid(-inputs)
                    layer['output'] = inputs

                # Compute loss
                output = inputs
                loss = np.mean((output - y) ** 2)
                total_loss += loss

                # Backward pass
                grad = 2 * (output - y)
                for layer in reversed(self.layers):
                    grad = self.backward(grad)

                # Update weights
                self.update_weights(learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(x_train)}")
            
            
    def predict(self, x_test):
        predictions = []
        for x in x_test:
            inputs = x
            for layer in self.layers:
                weights = layer['weights']
                biases = layer['biases']
                activation = layer['activation']
                inputs = np.dot(inputs, weights) + biases
                if activation == 'relu':
                    inputs = self.relu(inputs)
                elif activation == 'sigmoid':
                    inputs = self.sigmoid(-inputs)
            predictions.append(inputs.squeeze())
        return np.array(predictions)
