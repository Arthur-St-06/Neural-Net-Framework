#include <stdio.h>
#include <iostream>
#include <chrono>

#include "Model.cuh"

int main()
{
	Data<float>* data = new Data<float>;

	Model* model = new Model;

	model->Add(2, 64, "relu");
	model->Add(64, 64, "relu");
	model->Add(64, 3, "softmax", "categorical_crossentropy");

	model->Compile("adam");
	model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 1001, 100);
	
	model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

	return 0;
}