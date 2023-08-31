#include <stdio.h>

#include "Model.cuh"

int main()
{
	bool test = true;

	if (test == true)
	{
		Data<float>* data = new Data<float>;

		data->LoadValidatingDataInputs();

		Model<float>* model = new Model<float>;

		int neurons = 512;

		model->Add(784, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, 10, "softmax", "categorical_crossentropy");

		model->LoadFromFile();

		model->Compile("adam");

		model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());
	}
	else
	{
		Data<float>* data = new Data<float>;

		data->LoadTrainingDataInputs();
		data->LoadTrainingDataOutputs();
		data->LoadValidatingDataInputs();

		Model<float>* model = new Model<float>;

		int neurons = 512;

		model->Add(784, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, neurons, "relu");
		model->Add(neurons, 10, "softmax", "categorical_crossentropy");

		model->Compile("adam");

		model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 100, 10, 600);

		model->SaveToFile();

		model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());
	}
	

	return 0;
}