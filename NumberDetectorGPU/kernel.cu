#include <stdio.h>

#include "Model.cuh"

int main()
{
	// Flag to determine if the program should run in validation mode
	bool validate = true;

	// Create a Data object for managing input and output data for training and validating
	Data<float>* data = new Data<float>;

	// If validation mode is disabled, load training data inputs and outputs
	if (validate == false)
	{
		data->LoadTrainingDataInputs();
		data->LoadTrainingDataOutputs();
	}

	// Always load validating data inputs to test the model performace at the end
	data->LoadValidatingDataInputs();
	data->LoadValidatingDataOutputs();

	// Create a Model object for building and training the neural network
	Model<float>* model = new Model<float>;

	// Define the number of neurons in a hidden layer
	int neurons = 32;

	// Add layers to the neural network model
	model->Add(784, neurons, "relu");
	model->Add(neurons, neurons, "relu");
	model->Add(neurons, 10, "softmax", "categorical_crossentropy");

	// Compile the model using the Adam optimizer
	model->Compile("adam");

	// If in validation mode, load a pre-trained model
	if (validate == true)
	{
		model->LoadFromFile();
	}
	else
	{
		// Fit the model on the training data for 3 epochs with a batch size of 60 and printing every 10th epoch
		model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 3, 1, 64);
		// Save the trained model to a file
		model->SaveToFile();
	}

	// Test the model's performance on the validating data
	model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

	// Clean up allocated memory
	delete model;
	delete data;

	return 0;
}