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

		//cublasCreate(&handle);
	}

	void Add(int input, int output, std::string activation_function_type, std::string loss_type = "none", float weight_regularizer_l1 = 0.0f, float bias_regularizer_l1 = 0.0f, float weight_regularizer_l2 = 5e-4f, float bias_regularizer_l2 = 5e-4f)
	{
		m_activation_function_type = activation_function_type;
		m_loss_type = loss_type;

		if (m_activation_function_type == "relu")
		{
			DenseLayer<T>* dense = new DenseLayer<T>(input, output, INIT_TYPE::Xavier_Normal, weight_regularizer_l1, bias_regularizer_l1, weight_regularizer_l2, bias_regularizer_l2);

			Layer<T>* dense_layer = new Layer<T>(dense);

			m_layers.push_back(dense_layer);

			ActivationFunction<T>* activation_funciton = new ActivationFunction<T>(ACTIVATION_TYPE::Relu);
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
		else if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
		{
			DenseLayer<T>* dense = new DenseLayer<T>(input, output, INIT_TYPE::Xavier_Normal, 0.0f, 0.0f, 0.0f, 0.0f);

			Layer<T>* dense_layer = new Layer<T>(dense);

			m_layers.push_back(dense_layer);

			SoftmaxCategoricalCrossentropy<T>* activation_funciton = new SoftmaxCategoricalCrossentropy<T>();
			Layer<T>* activation_function_layer = new Layer<T>(activation_funciton);
			m_layers.push_back(activation_function_layer);
		}
	}

	void Compile(std::string optimizer_type, float learning_rate = 0.02f, float decay = 1e-3f, float epsilon = 1e-7f, float beta_1 = 0.9f, float beta_2 = 0.999f)
	{
		m_optimizer_type = optimizer_type;

		for (int i = 0; i < m_layers.size() - 1; i += 2)
		{
			if (m_optimizer_type == "adam")
			{
				Adam<T>* adam_optimizer = new Adam<T>(learning_rate, decay, epsilon, beta_1, beta_2);
				Optimizer<T>* optimizer = new Optimizer<T>(adam_optimizer);
				m_optimizers.push_back(optimizer);
			}
		}
	}

	void Fit(std::vector<std::vector<T>>* data_inputs, std::vector<std::vector<T>>* data_outputs, size_t epochs, size_t print_every, size_t batch_size = 1)
	{
		// Setting batched data
		size_t amount_of_batches = std::floor(data_inputs[0].size() / batch_size);

		for (size_t step = 0; step < amount_of_batches; step++)
		{
			std::vector<std::vector<T>> batch_data_inputs_vector;
			std::vector<T> batch_data_outputs_vector;
		
			for (size_t j = 0; j < batch_size; j++)
			{
				batch_data_inputs_vector.push_back(data_inputs[0][step * batch_size + j]);
				batch_data_outputs_vector.push_back(data_outputs[0][0][step * batch_size + j]);
			}
		
			Matrix<T>* batch_data_inputs_matrix = new Matrix<T>(batch_data_inputs_vector);
			Matrix<T>* batch_data_outputs_matrix = new Matrix<T>(batch_data_outputs_vector);
		
			m_data_inputs.push_back(batch_data_inputs_matrix);
			m_data_outputs.push_back(batch_data_outputs_matrix);
		}

		SetInputs();

		StartTimer();

		float reg_loss = 0.0f;

		float current_loss = 0.0f;
		float current_accuracy = 0.0f;

		float accumulated_loss = 0.0f;
		float accumulated_accuracy = 0.0f;

		//Data<float>* data = new Data<float>;
		//
		//data->LoadValidatingDataInputs();

		for (size_t epoch = 0; epoch < epochs; epoch++)
		{
			for (size_t step = 0; step < amount_of_batches; step++)
			{
				m_layers[0]->GetDenseLayer()->Forward(m_data_inputs[step]);
				m_layers[1]->GetActivationFunction()->Forward();

				// Fix add if statements to allow for non softmax categorical crossentropy last layer
				for (size_t i = 2; i < m_layers.size() - 3; i += 2)
				{
					m_layers[i]->GetDenseLayer()->Forward();
					m_layers[i + 1]->GetActivationFunction()->Forward();
				}

				m_layers[m_layers.size() - 2]->GetDenseLayer()->Forward();
				m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->Forward(m_data_outputs[step]);

				if (epoch % print_every == 0)
				{
					reg_loss = 0;
					// Do not count last dense layer
					for (size_t i = 0; i < m_layers.size() - 2; i += 2)
					{
						reg_loss += m_layers[i]->GetDenseLayer()->RegularizationLoss();
					}

					current_loss = m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetLoss();

					if (step == 0)
					{
						accumulated_loss = 0.0f;
						accumulated_accuracy = 0.0f;
					}

					current_loss = m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetLoss();
					current_accuracy = m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetAccuracy();

					accumulated_loss += current_loss;
					accumulated_accuracy += current_accuracy;

					std::cout << "Step: " << step + 1;
					std::cout << ", loss: " << current_loss;
					std::cout << ", reg loss: " << reg_loss;
					std::cout << ", accuracy: " << current_accuracy << std::endl;
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
			if (epoch % print_every == 0)
			{
				std::cout << "Epoch: " << epoch;
				std::cout << ", loss: " << accumulated_loss / amount_of_batches;
				std::cout << ", accuracy: " << accumulated_accuracy / amount_of_batches << std::endl << std::endl;
			}

			//ClearData();
			//
			//Test(data->GetValidatingDataInputs(), data->GetValidatingDataOutputs());
			//
			//ClearData();
		}

		ClearData();

		StopTimer();
	}

	void Test(std::vector<std::vector<T>>* testing_data_inputs, std::vector<std::vector<T>>* testing_data_outputs)
	{
		Matrix<T>* inputs_matrix = new Matrix<T>(testing_data_inputs);
		Matrix<T>* outputs_matrix = new Matrix<T>(testing_data_outputs);

		m_data_inputs.push_back(inputs_matrix);
		m_data_outputs.push_back(outputs_matrix);
	
		SetInputs();
	
		m_layers[0]->GetDenseLayer()->Forward(m_data_inputs[0]);
		m_layers[1]->GetActivationFunction()->Forward();

		// Fix add if statements to allow for non softmax categorical crossentropy last layer
		for (size_t i = 2; i < m_layers.size() - 3; i += 2)
		{
			m_layers[i]->GetDenseLayer()->Forward();
			m_layers[i + 1]->GetActivationFunction()->Forward();
		}

		m_layers[m_layers.size() - 2]->GetDenseLayer()->Forward();
		m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->Forward(m_data_outputs[0]);
	
		std::cout << "Loss: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetLoss();
		std::cout << ", accuracy: " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetAccuracy();
		std::cout << ", predictions: ";

		for (size_t i = 0; i < 10; i++)
		{
			std::cout << " " << m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->GetLoss()->GetPredictions()[i];
		}

		std::cout << std::endl;

		ClearData();
	}

	void ClearData()
	{
		for (size_t i = 0; i < m_data_inputs.size(); i++)
		{
			delete m_data_inputs[i];
			delete m_data_outputs[i];
		}

		m_data_inputs.clear();
		m_data_outputs.clear();
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

	std::vector<Matrix<T>*> m_data_inputs;
	std::vector<Matrix<T>*> m_data_outputs;

	std::string m_activation_function_type;
	std::string m_loss_type;
	std::string m_optimizer_type;

	std::vector<std::vector<std::vector<float>>> m_parameters;

	// Timer
	std::chrono::high_resolution_clock::time_point m_timer_begin;
	std::chrono::high_resolution_clock::time_point m_timer_end;
	std::chrono::nanoseconds m_timer_elapsed;

	//cublasHandle_t handle;

	void SetInputs()
	{
		SetLayersInputs();
		SetOptimizersInputs();
	}

	void SetLayersInputs()
	{
		m_layers[0]->GetDenseLayer()->SetInputs(m_data_inputs[0]);


		for (int i = 1; i < m_layers.size() - 1; i += 2)
		{
			m_layers[i]->GetActivationFunction()->SetInputs(m_layers[i - 1]->GetDenseLayer()->GetOutputs());
			m_layers[i + 1]->GetDenseLayer()->SetInputs(m_layers[i]->GetActivationFunction()->GetOutputs());
		}

		if (m_activation_function_type == "softmax" && m_loss_type == "categorical_crossentropy")
			m_layers[m_layers.size() - 1]->GetSoftmaxCategoricalCrossentropy()->SetInputs(m_layers[m_layers.size() - 2]->GetDenseLayer()->GetOutputs(), m_data_outputs[0]);
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