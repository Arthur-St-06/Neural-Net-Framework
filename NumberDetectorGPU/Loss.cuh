#pragma once

#include "Matrix.cuh"

// Loss is a parent class for CategoricalCrossentropyLoss as it simplifies addition of new Loss types
template <typename T>
class Loss
{
public:
	Loss() {	}

	// Calculate both loss and accuracy based on ground truth
	void Calculate(Matrix<T>* ground_truth)
	{
		m_ground_truth = ground_truth;

		// Find loss
		Forward();
		m_data_loss = m_negative_log_confidencies->Mean();

		// Find accuracy
		m_num_predictions_equal_ground_truth->RowArgmax(m_predictions);
		m_num_predictions_equal_ground_truth->CompareMatrixAndVector(m_num_predictions_equal_ground_truth, m_ground_truth);
		m_data_accuracy = m_num_predictions_equal_ground_truth->ColMean();
	}

	// Get a pointer to the predictions (output values)
	T* GetPredictions()
	{
		T* predictions = new T[m_predictions->GetRow()];

		cudaMemcpy(predictions, m_predictions->GetMatrix(), m_predictions->GetRow() * sizeof(T), cudaMemcpyDeviceToHost);

		return predictions;
	}

	T GetLoss()
	{
		return m_data_loss;
	}

	T GetAccuracy()
	{
		return m_data_accuracy;
	}

	Matrix<T>* GetDinputs()
	{
		return m_dinputs;
	}

private:
	virtual void Forward() {}
	virtual void Backward() {}

protected:
	Matrix<T>* m_predictions;
	Matrix<T>* m_ground_truth;
	Matrix<T>* m_one_hot_encoded_ground_truth;

	// Loss
	Matrix<T>* m_negative_log_confidencies;
	T m_data_loss;

	// Accuracy
	Matrix<T>* m_num_predictions_equal_ground_truth;
	T m_data_accuracy;

	// Backpropagation
	Matrix<T>* m_dinputs;
};

template <class T>
class CategoricalCrossentropyLoss : public Loss<T>
{
public:
	CategoricalCrossentropyLoss() : Loss<T>()
	{
		// Initialize empty matricies, which will be filled in SetInputs functions
		m_ground_truth = new Matrix<T>;
		m_negative_log_confidencies = new Matrix<T>;
		m_num_predictions_equal_ground_truth = new Matrix<T>;
		m_one_hot_encoded_ground_truth = new Matrix<T>;

		m_clipped_predictions = new Matrix<T>;
		m_dinputs = new Matrix<T>;
	}

	// Set inputs (predictions and ground truth) for the loss layer
	void SetInputs(Matrix<T>* predictions, Matrix<T>* ground_truth)
	{
		if (m_ground_truth->Cleared() == false)
		{
			m_negative_log_confidencies->Clear();
			m_num_predictions_equal_ground_truth->Clear();
			m_one_hot_encoded_ground_truth->Clear();

			m_clipped_predictions->Clear();
			m_dinputs->Clear();
		}

		if (m_predictions != predictions)
			m_predictions = predictions;

		if (m_ground_truth != ground_truth)
			m_ground_truth = ground_truth;

		m_negative_log_confidencies->InitMatrix(1, m_ground_truth->GetRow());
		m_num_predictions_equal_ground_truth->InitMatrix(m_ground_truth->GetRow(), 1);

		if (m_ground_truth->GetCol() == 1)
			m_one_hot_encoded_ground_truth->InitMatrix(m_ground_truth->GetRow(), m_predictions->GetRow());

		m_clipped_predictions->InitMatrix(m_predictions->GetCol(), m_predictions->GetRow());
		m_dinputs->InitMatrix(m_predictions->GetCol(), m_predictions->GetRow());
	}

private:
	Matrix<T>* m_clipped_predictions;

	// Forward pass to calculate loss
	void Forward()
	{
		m_clipped_predictions->Clip(m_predictions, 1e-7, 1 - 1e-7);
		m_negative_log_confidencies->GetValuesAccordingToMatrices(m_clipped_predictions, m_ground_truth);
		m_negative_log_confidencies->NegativeLog(m_negative_log_confidencies);
	}

	// Perform the backward pass to compute gradients
	void Backward()
	{
		if (m_ground_truth->GetCol() == 1)
			m_one_hot_encoded_ground_truth->OneHotEncode(m_ground_truth);

		// Calculate gradient
		m_dinputs->DivideMatrices(m_one_hot_encoded_ground_truth, m_predictions);

		m_dinputs->SubstractValueFromMatrix(m_dinputs, 0);
		// Normalize gradient for optimizer as it sums all will sum all of the samples to one
		m_dinputs->DivideMatrixByValue(m_dinputs, m_dinputs->GetCol());
	}
};