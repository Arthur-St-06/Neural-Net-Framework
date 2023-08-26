#pragma once

#include "Matrix.cuh"

class Loss
{
public:
	Matrix<half>* m_predictions;
	Matrix<half>* m_ground_truth;
	Matrix<half>* m_one_hot_encoded_ground_truth;

	// Loss
	Matrix<half>* m_negative_log_confidencies;
	float m_data_loss;

	// Accuracy
	Matrix<half>* m_num_predictions_equal_ground_truth;
	float m_data_accuracy;

	Loss()
	{	}

	void Calculate()
	{
		// Find loss
		Forward();
		m_data_loss = m_negative_log_confidencies->Mean();

		// Find accuracy
		m_num_predictions_equal_ground_truth->RowArgmax(m_predictions);
		m_num_predictions_equal_ground_truth->CompareMatrixAndVector(m_num_predictions_equal_ground_truth, m_ground_truth);
		m_data_accuracy = m_num_predictions_equal_ground_truth->ColMean();
	}

	half* GetPredictions()
	{
		half* predictions = new half[3];

		cudaMemcpy(predictions, m_predictions->GetMatrix(), 3 * sizeof(half), cudaMemcpyDeviceToHost);

		return predictions;
	}

	float GetLoss()
	{
		return m_data_loss;
	}

	float GetAccuracy()
	{
		return m_data_accuracy;
	}

	Matrix<half>* GetDinputs()
	{
		return m_dinputs;
	}

private:
	virtual void Forward() {}
	virtual void Backward() {}

protected:
	

	// Backpropagation
	Matrix<half>* m_dinputs;
};

template <class T>
class CategoricalCrossentropyLoss : public Loss
{
public:
	CategoricalCrossentropyLoss() : Loss()
	{
		// Initialize empty matricies, which will be filled in SetInputs functions
		m_ground_truth = new Matrix<T>;
		m_negative_log_confidencies = new Matrix<T>;
		m_num_predictions_equal_ground_truth = new Matrix<T>;
		m_one_hot_encoded_ground_truth = new Matrix<T>;

		m_clipped_predictions = new Matrix<T>;
		m_dinputs = new Matrix<T>;
	}

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

	void Forward()
	{
		m_clipped_predictions->Clip(m_predictions, 1e-7, 1 - 1e-7);
		m_negative_log_confidencies->GetValuesAccordingToMatrices(m_clipped_predictions, m_ground_truth);
		m_negative_log_confidencies->NegativeLog(m_negative_log_confidencies);
	}
};