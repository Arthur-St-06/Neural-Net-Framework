#pragma once

#include "Matrix.cuh"

#include <fstream>
#include <sstream>

template <class T>
class Data
{
public:
	Data()
	{
		m_training_data_inputs = new std::vector<std::vector<T>>;
		m_training_data_outputs = new std::vector<std::vector<T>>;

		m_validating_data_inputs = new std::vector<std::vector<T>>;
		m_validating_data_outputs = new std::vector<std::vector<T>>;
	}

	~Data()
	{
		delete m_training_data_inputs;
		delete m_training_data_outputs;

		delete m_validating_data_inputs;
		delete m_validating_data_outputs;
	}

	// Loads training data inputs from a file
	void LoadTrainingDataInputs()
	{
		std::ifstream inputFile("TrainingDataInputs.txt");

		std::vector<float> current_data;
		std::string line;

		while (std::getline(inputFile, line)) {
			if (!line.empty()) {
				std::istringstream iss(line);
				float pixelValue;

				while (iss >> pixelValue) {
					current_data.push_back(pixelValue);
				}
			}
			else {
				m_training_data_inputs[0].push_back(current_data);
				current_data.clear();
			}
		}

		inputFile.close();
	}

	// Loads training data outputs from a file
	void LoadTrainingDataOutputs()
	{
		std::ifstream inputFile("TrainingDataOutputs.txt");

		std::vector<float> current_data;
		std::string line;

		while (std::getline(inputFile, line)) {
			std::istringstream iss(line);
			float pixelValue;

			while (iss >> pixelValue) {
				current_data.push_back(pixelValue);
			}
		}

		m_training_data_outputs[0].push_back(current_data);

		inputFile.close();
	}

	// Loads validating data inputs from a file
	void LoadValidatingDataInputs()
	{
		std::ifstream inputFile("ValidatingDataInputs.txt");

		std::vector<float> current_data;
		std::string line;

		while (std::getline(inputFile, line)) {
			std::istringstream iss(line);
			float pixelValue;

			while (iss >> pixelValue) {
				current_data.push_back(pixelValue);
			}
		}

		m_validating_data_inputs[0].push_back(current_data);

		inputFile.close();
	}

	// Loads validating data outputs from a file
	void LoadValidatingDataOutputs()
	{
		std::ifstream inputFile("ValidatingDataOutputs.txt");

		std::vector<float> current_data;
		std::string line;

		while (std::getline(inputFile, line)) {
			std::istringstream iss(line);
			float pixelValue;

			while (iss >> pixelValue) {
				current_data.push_back(pixelValue);
			}
		}

		m_validating_data_outputs[0].push_back(current_data);

		inputFile.close();
	}

	std::vector<std::vector<T>>* GetTrainingDataInputs() { return m_training_data_inputs; }
	std::vector<std::vector<T>>* GetTrainingDataOutputs() { return m_training_data_outputs; }
	std::vector<std::vector<T>>* GetValidatingDataInputs() { return m_validating_data_inputs; }
	std::vector<std::vector<T>>* GetValidatingDataOutputs() { return m_validating_data_outputs; }

private:
	std::vector<std::vector<T>>* m_training_data_inputs;
	std::vector<std::vector<T>>* m_training_data_outputs;

	std::vector<std::vector<T>>* m_validating_data_inputs;
	std::vector<std::vector<T>>* m_validating_data_outputs;
};