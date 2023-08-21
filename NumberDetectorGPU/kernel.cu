#include <stdio.h>

#include "Model.cuh"

int main()
{
	Data<half>* data = new Data<half>;

	Model<half>* model = new Model<half>;

	model->Add(2, 8096, "relu");
	model->Add(8096, 8096, "relu");
	model->Add(8096, 3, "softmax", "categorical_crossentropy");

	model->Compile("adam");
	model->Fit(data->GetTrainingDataInputs(), data->GetTrainingDataOutputs(), 1001, 100);
	
	model->Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());

	return 0;
}