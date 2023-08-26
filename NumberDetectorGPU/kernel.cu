#include <stdio.h>

#include "Model.cuh"

int main()
{
	Data<half>* data = new Data<half>;

	Model<half>* model = new Model<half>;

	model->Add(2, 64, "relu");
	model->Add(64, 3, "softmax", "categorical_crossentropy");

	//model->SaveToFile();
	model->LoadFromFile();

	model->Compile("adam");

	model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 1000, 100);

	model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

	return 0;
}