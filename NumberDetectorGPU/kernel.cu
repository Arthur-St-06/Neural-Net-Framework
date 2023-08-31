#include <stdio.h>

#include "Model.cuh"

int main()
{
	Data<float>* data = new Data<float>;

	Model<float>* model = new Model<float>;

	model->Add(2, 128, "relu");
	model->Add(128, 128, "relu");
	model->Add(128, 3, "softmax", "categorical_crossentropy");

	//model->LoadFromFile();
	//model->SaveToFile();

	model->Compile("adam");

	model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 1001, 100, 3000);

	//model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

	return 0;
}