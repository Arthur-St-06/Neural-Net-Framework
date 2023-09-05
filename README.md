# Neural Network Framework in C++ with GPU Acceleration

## Overview

This project is a neural network framework implemented from scratch in C++ with a focus on object-oriented programming and GPU acceleration. It includes custom implementations of key components of neural networks, such as dense layers, activation functions (Sigmoid and ReLU), loss functions (Mean Squared Error), and optimization algorithms (SGD, RMSprop, and Adam).

The code is designed to demonstrate a fundamental understanding of neural networks and GPU programming. It showcases various aspects, including:

- Creating neural network layers using object-oriented programming principles.
- Implementing activation functions (ReLU and Softmax) for non-linear transformations.
- Defining loss functions (Categorical Cross-Entropy) to measure the network's performance.
- Incorporating optimization algorithms (SGD, RMSprop, and Adam) for efficient training.
- Making AI models to generalize effectively using reguralization losses of weights and biases.
- Leveraging GPU acceleration for matrix computations (CUDA-based).

## System Configuration

On my development machine, which runs Windows 10, I have the following hardware, making the training time of the model provided below equal to 2.142 seconds:

- CPU: Intel Core i9-13900K
- GPU: NVIDIA RTX 4090

## Usage

To use this neural network framework, follow these steps:

1. Download the repository to your local machine.
2. Compile the code using your preferred C++ compiler with CUDA support.
3. Use GPU acceleration for matrix computations (requires a compatible NVIDIA GPU and CUDA toolkit).
4. Modify and extend the code to build your custom neural network architecture or experiment with different configurations.

## Sample Code

### Training the model

1. Set training data inputs in form of 2D arrays in the TrainingDataInputs.txt file (currently project has MNIST dataset loaded in).
2. Set training data outputs (1 output per row) in TrainingDataOutputs.txt.
3. Create Data object and call Load functions for both input and output training data.

```cuda

Data<float>* data = new Data<float>;
data->LoadTrainingDataInputs();
data->LoadTrainingDataOutputs();

```

4. Create model object.

```cuda

Model<float>* model = new Model<float>;

```

5. Add layers to the model.

```cuda

// Each layer in the model should be added in the following way
// model->Add(number of input neurons, number of output neurons, activation function, loss type(none is default),
// weight reguralizer l1(0.0f), bias reguralizer l1(0.0f), weight reguralizer l2(0.0f), bias reguralizer l2(0.0f));

// Data from the TrainingDataInputs.txt file is automatically flattened

// Example layers
model->Add(784, 32, "relu");
model->Add(32, 32, "relu");
model->Add(32, 10, "softmax", "categorical_crossentropy");

```

6. Set model's optimizer (choose from SGD, RMSprop or Adam, and set appropriate arguments for the chosen optimizer).

```cuda

model->Compile("adam");

```

7. Fit the model.

```cuda
// model->Fit(input data, output data, epochs, print every epoch, batch)

model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 3, 1, 64);

```

### Validate the model

1. Set validating data inputs in form of 2D arrays in the ValidatingDataInputs.txt file (currently project has number drawn in Paint and converted to numbers loaded in).
2. Set validating data outputs in the ValidatingDataOuputs.txt file.
3. Create Data object and call Load functions for both input and output validation data.

```cuda

Data<float>* data = new Data<float>;
data->LoadValidatingDataInputs();
data->LoadValidatingDataOutputs();

```

4. Create model object.

```cuda

Model<float>* model = new Model<float>;

```

5. Add layers to the model.

```cuda

// The same layers should be added to the validating model as to the training model.
model->Add(784, 32, "relu");
model->Add(32, 32, "relu");
model->Add(32, 10, "softmax", "categorical_crossentropy");

```

6. Load the weights and biases of the trained model.

```cuda

model->LoadFromFile();

```

7. Test model to get loss, accuracy, predicted classes probabilities and predicted class.

```cuda

model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

```
