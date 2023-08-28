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

#include <fstream>
#include <sstream>

template <class T>
class Model
{
public:
	Model()
	{
		InitTimer();
	}

	void AddTmp(DenseLayer<T> dense, std::string activation_function_type, std::string loss_type = "none")
	{
		m_activation_function_type = activation_function_type;
		m_loss_type = loss_type;

		Layer<T>* dense_layer = new Layer<T>(&dense);

		m_layers.push_back(dense_layer);

		if (m_activation_function_type == "relu")
		{
			ActivationFunction<T>* activation_funciton = new ActivationFunction<T>(ACTIVATION_TYPE::Relu);
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
		else if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
		{
			SoftmaxCategoricalCrossentropy<T>* activation_funciton = new SoftmaxCategoricalCrossentropy<T>();
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
	}

	void Add(int input, int output, std::string activation_function_type, std::string loss_type = "none")
	{
		m_activation_function_type = activation_function_type;
		m_loss_type = loss_type;

		if (m_activation_function_type == "relu")
		{
			DenseLayer<T>* dense = new DenseLayer<T>(input, output, INIT_TYPE::Xavier_Normal, 0, 0, 5e-4, 5e-4);

			Layer<T>* dense_layer = new Layer<T>(dense);

			m_layers.push_back(dense_layer);

			ActivationFunction<T>* activation_funciton = new ActivationFunction<T>(ACTIVATION_TYPE::Relu);
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
		else if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
		{
			DenseLayer<T>* dense = new DenseLayer<T>(input, output, INIT_TYPE::Xavier_Normal, 0, 0, 0, 0);

			Layer<T>* dense_layer = new Layer<T>(dense);

			m_layers.push_back(dense_layer);

			SoftmaxCategoricalCrossentropy<T>* activation_funciton = new SoftmaxCategoricalCrossentropy<T>();
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
	}

	void Compile(SGD sgd)
	{
		m_optimizer_type = "sgd";

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_optimizers.push_back(new Optimizer<T>(&sgd));
		}
	}

	void Compile(RMSprop rmsprop)
	{
		m_optimizer_type = "rmsprop";

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_optimizers.push_back(new Optimizer<T>(&rmsprop));
		}
	}

	void Compile(std::string optimizer_type)
	{
		m_optimizer_type = optimizer_type;

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			if (m_optimizer_type == "adam")
			{
				Adam<T>* adam_optimizer = new Adam<T>();
				Optimizer<T>* optimizer = new Optimizer<T>(adam_optimizer);
				m_optimizers.push_back(optimizer);
			}
		}
	}

	void TmpFit(Matrix<T> data_inputs, Matrix<T> data_outputs, size_t epochs, size_t print_every)
	{
		m_data_inputs = &data_inputs;
		m_data_outputs = &data_outputs;

		SetInputs();
	}

	void Fit(Matrix<T> data_inputs, Matrix<T> data_outputs, size_t epochs, size_t print_every)
	{
		m_data_inputs = &data_inputs;
		m_data_outputs = &data_outputs;

		SetInputs();

		float reg_loss = 0.0f;

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

	void Test(Matrix<T> testing_data_inputs, Matrix<T> testing_data_outputs)
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
		std::cout << ", predictions: " << __half2float(m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[0]);
		std::cout << " " << __half2float(m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[1]);
		std::cout << " " << __half2float(m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[2]) << std::endl;
	}

	std::vector<std::vector<std::vector<float>>> Save()
	{
		m_parameters.clear();
		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_parameters.push_back(m_layers[i]->GetDenseLayer()->GetWeights()->GetVectorMatrix());
			m_parameters.push_back(m_layers[i]->GetDenseLayer()->GetBiases()->GetVectorMatrix());
		}

		return m_parameters;
	}

	void SaveToFile()
	{
		m_parameters.clear();
		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_parameters.push_back(m_layers[i]->GetDenseLayer()->GetWeights()->GetVectorMatrix());
			m_parameters.push_back(m_layers[i]->GetDenseLayer()->GetBiases()->GetVectorMatrix());
		}

		std::ofstream outFile("Datatxt.txt");
		if (outFile.is_open()) {
			for (const auto& vectorOfVectors : m_parameters) {
				for (const auto& innerVector : vectorOfVectors) {
					for (const float value : innerVector) {
						outFile << value << " ";
					}
					outFile << std::endl;
				}
				outFile << "---" << std::endl;  // Separate different vector of vectors
			}
			outFile.close();
			std::cout << "Data saved to data.txt" << std::endl;
		}
		else {
			std::cerr << "Unable to open file for writing." << std::endl;
		}
	}

	void Load(std::vector<std::vector<std::vector<float>>> parameters)
	{
		m_parameters = parameters;

		SetWeights();
	}

	void LoadFromFile()
	{
		std::ifstream inFile("Datatxt.txt");
		if (inFile.is_open()) {
			m_parameters.clear();
			std::vector<std::vector<float>> currentVectorOfVectors;
			std::vector<float> currentInnerVector;
			std::string line;
			while (std::getline(inFile, line)) {
				if (line == "---") {
					m_parameters.push_back(currentVectorOfVectors);
					currentVectorOfVectors.clear();
				}
				else {
					std::istringstream iss(line);
					float value;
					while (iss >> value) {
						currentInnerVector.push_back(value);
					}
					currentVectorOfVectors.push_back(currentInnerVector);
					currentInnerVector.clear();
				}
			}
			inFile.close();
		}
		else {
			std::cerr << "Unable to open file for reading." << std::endl;
		}

		SetWeights();
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
	std::vector<Layer<T>*> m_layers;
	std::vector<Optimizer<T>*> m_optimizers;

	Matrix<T>* m_data_inputs;
	Matrix<T>* m_data_outputs;

	std::string m_activation_function_type;
	std::string m_loss_type;
	std::string m_optimizer_type;

	std::vector<std::vector<std::vector<float>>> m_parameters;

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

	void SetWeights()
	{
		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			m_layers[i]->GetDenseLayer()->SetParameters(&m_parameters[i], &m_parameters[i + 1]);
		}
	}
};