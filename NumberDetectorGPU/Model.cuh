#include <stdio.h>
#include <iostream>
#include<vector>

#include "Matrix.cuh"
#include "DenseLayer.cuh"
#include "ActivationFunctions.cuh"
#include "Loss.cuh"
#include "SoftmaxCategoricalCrossentropy.cuh"
#include "Optimizer.cuh"
#include "Data.cuh"
#include "Layer.cuh"

class Model
{
public:
	Model()
	{

	}

	void Add(DenseLayer<float>* dense_layer, std::string activation_function_type, std::string loss_type = "none")
	{
		Layer<float>* layer = new Layer<float>(dense_layer);

		if (activation_function_type == "relu")
		{
			ActivationFunction<float>* activation_funciton = new ActivationFunction<float>(dense_layer->GetOutputs(), ACTIVATION_TYPE::Relu);
		}
		else if (activation_function_type == "softmax" && loss_type == "categorical_crossentropy")
		{
			//ActivationFunction<float>* activation_funciton = new ActivationFunction<float>(dense_layer->GetOutputs(), ACTIVATION_TYPE::Relu);
		}

		m_layers.push_back(layer);
	}

	void Add(ActivationFunction<float>* activation_function)
	{
		Layer<float>* layer = new Layer<float>(activation_function);

		m_layers.push_back(layer);
	}

	void Add(SoftmaxCategoricalCrossentropy<float>* softmax_categorical_crossentropy)
	{
		Layer<float>* layer = new Layer<float>(softmax_categorical_crossentropy);

		m_layers.push_back(layer);
	}

	void Compile()
	{

	}

	void Fit(Matrix<float>* validating_data_inputs, Matrix<float>* validating_data_outputs)
	{
		SetInputs(validating_data_inputs, validating_data_outputs);
	}

private:
	std::vector<Layer<float>*> m_layers;
	std::vector<Data<float>*> data_vector;
	std::vector<DenseLayer<float>*> dense_layer_vector;
	std::vector<ActivationFunction<float>*> activation_function_vector;
	std::vector<SoftmaxCategoricalCrossentropy<float>*> softmax_categorical_crossentropy_vector;

	void SetInputs(Matrix<float>* validating_data_inputs, Matrix<float>* validating_data_outputs)
	{
		m_layers[0]->GetDenseLayer()->SetInputs(validating_data_inputs);


		for (int i = 0; i < dense_layer_vector.size(); i++)
		{
			
		}

		//SoftmaxLoss.SetInputs(dense3.GetOutputs(), data.GetValidatingDataOutputs());
	}
};