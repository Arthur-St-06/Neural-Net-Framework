#include <stdio.h>
#include <iostream>
#include<vector>
#include <chrono>

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
		InitTimer();
	}

	void AddTmp(DenseLayer<float> dense, std::string activation_function_type, std::string loss_type = "none")
	{
		m_activation_function_type = activation_function_type;
		m_loss_type = loss_type;

		Layer<float>* dense_layer = new Layer<float>(&dense);

		m_layers.push_back(dense_layer);

		if (m_activation_function_type == "relu")
		{
			ActivationFunction<float>* activation_funciton = new ActivationFunction<float>(ACTIVATION_TYPE::Relu);
			Layer<float>* activation_function_layer = new Layer<float>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
		else if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
		{
			SoftmaxCategoricalCrossentropy<float>* activation_funciton = new SoftmaxCategoricalCrossentropy<float>();
			Layer<float>* activation_function_layer = new Layer<float>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
	}

	void Add(int input, int output, std::string activation_function_type, std::string loss_type = "none")
	{
		m_activation_function_type = activation_function_type;
		m_loss_type = loss_type;

		DenseLayer<float>* dense = new DenseLayer<float>(input, output);

		Layer<float>* dense_layer = new Layer<float>(dense);

		m_layers.push_back(dense_layer);

		if (m_activation_function_type == "relu")
		{
			ActivationFunction<float>* activation_funciton = new ActivationFunction<float>(ACTIVATION_TYPE::Relu);
			Layer<float>* activation_function_layer = new Layer<float>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
		else if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
		{
			SoftmaxCategoricalCrossentropy<float>* activation_funciton = new SoftmaxCategoricalCrossentropy<float>();
			Layer<float>* activation_function_layer = new Layer<float>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
	}

	void Compile(SGD sgd)
	{
		m_optimizer_type = "sgd";

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_optimizers.push_back(new Optimizer(&sgd));
		}
	}

	void Compile(RMSprop rmsprop)
	{
		m_optimizer_type = "rmsprop";

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_optimizers.push_back(new Optimizer(&rmsprop));
		}
	}

	void Compile(std::string optimizer_type)
	{
		m_optimizer_type = optimizer_type;

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			if (m_optimizer_type == "adam")
			{
				Adam* adam_optimizer = new Adam();
				Optimizer* optimizer = new Optimizer(adam_optimizer);
				m_optimizers.push_back(optimizer);
			}
		}
	}

	void TmpFit(Matrix<float> data_inputs, Matrix<float> data_outputs, size_t epochs, size_t print_every)
	{
		m_data_inputs = &data_inputs;
		m_data_outputs = &data_outputs;

		SetInputs();
	}

	void Fit(Matrix<float> data_inputs, Matrix<float> data_outputs, size_t epochs, size_t print_every)
	{
		m_data_inputs = &data_inputs;
		m_data_outputs = &data_outputs;

		SetInputs();

		float reg_loss;

		StartTimer();

		for (size_t epoch = 0; epoch < epochs; epoch++)
		{
			// Fix add if statements to allow for non softmax categorical crossentropy last layer
			for (size_t i = 0; i < m_layers.size() - 3; i += 2)
			{
				m_layers[i]->GetDenseLayer()->Forward();
				m_layers[i + 1]->GetActivationFunction()->Forward();
			}
			m_layers[m_layers.size() - 2]->GetDenseLayer()->Forward();
			m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->Forward();

			if (epoch % print_every == 0)
			{
				reg_loss = 0;
				// Do not count last dense layer
				for (size_t i = 0; i < m_layers.size() - 2; i += 2)
				{
					reg_loss += m_layers[i]->GetDenseLayer()->RegularizationLoss();
				}

				std::cout << "Epoch: " << epoch;
				std::cout << ", loss: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetLoss();
				std::cout << ", reg loss: " << reg_loss;
				std::cout << ", accuracy: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetAccuracy() << std::endl;
			}

			m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->Backward();
			m_layers[m_layers.size() - 2]->GetDenseLayer()->Backward(m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetDinputs());

			for (int i = m_layers.size() - 3; i >= 0; i -= 2)
			{
				m_layers[i]->GetActivationFunction()->Backward(m_layers[i + 1]->GetDenseLayer()->GetDinputs());
				m_layers[i - 1]->GetDenseLayer()->Backward(m_layers[i]->GetActivationFunction()->GetDinputs());
			}


			for (size_t i = 0; i < m_optimizers.size(); i++)
			{
				m_optimizers[i]->GetAdam()->UpdateParams();
			}
		}

		StopTimer();
	}

	void Test(Matrix<float> testing_data_inputs, Matrix<float> testing_data_outputs)
	{
		m_data_inputs = &testing_data_inputs;
		m_data_outputs = &testing_data_outputs;

		SetInputs();

		for (size_t i = 0; i < m_layers.size() - 3; i += 2)
		{
			m_layers[i]->GetDenseLayer()->Forward();
			m_layers[i + 1]->GetActivationFunction()->Forward();
		}
		m_layers[m_layers.size() - 2]->GetDenseLayer()->Forward();
		m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->Forward();

		std::cout << "Loss: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetLoss();
		std::cout << ", accuracy: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetAccuracy();
		std::cout << ", predictions: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[0];
		std::cout << " " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[1];
		std::cout << " " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[2] << std::endl;;
	}

	void InitTimer()
	{
		m_timer_begin = std::chrono::high_resolution_clock::now();
		m_timer_end = std::chrono::high_resolution_clock::now();
		m_timer_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(m_timer_begin - m_timer_end);
	}

	void StartTimer()
	{
		m_timer_begin = std::chrono::high_resolution_clock::now();
	}

	void StopTimer()
	{
		m_timer_end = std::chrono::high_resolution_clock::now();
		m_timer_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(m_timer_end - m_timer_begin);

		printf("Fitting time: %.3f seconds.\n", m_timer_elapsed.count() * 1e-9);
	}

private:
	std::vector<Layer<float>*> m_layers;
	std::vector<Optimizer*> m_optimizers;

	Matrix<float>* m_data_inputs;
	Matrix<float>* m_data_outputs;

	std::string m_activation_function_type;
	std::string m_loss_type;
	std::string m_optimizer_type;

	// Timer
	std::chrono::high_resolution_clock::time_point m_timer_begin;
	std::chrono::high_resolution_clock::time_point m_timer_end;
	std::chrono::nanoseconds m_timer_elapsed;

	void SetInputs()
	{
		SetLayersInputs();
		SetOptimizersInputs();
	}

	void SetLayersInputs()
	{
		m_layers[0]->GetDenseLayer()->SetInputs(m_data_inputs);


		for (int i = 1; i < m_layers.size() - 1; i += 2)
		{
			m_layers[i]->GetActivationFunction()->SetInputs(m_layers[i - 1]->GetDenseLayer()->GetOutputs());
			m_layers[i + 1]->GetDenseLayer()->SetInputs(m_layers[i]->GetActivationFunction()->GetOutputs());
		}

		if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
			m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->SetInputs(m_layers[m_layers.size() - 2]->GetDenseLayer()->GetOutputs(), m_data_outputs);
	}

	void SetOptimizersInputs()
	{
		for (int i = 0; i < m_optimizers.size(); i++)
		{
			if (m_optimizer_type == "adam")
				m_optimizers[i]->GetAdam()->SetInputs(m_layers[i * 2]->GetDenseLayer());
		}
	}
};