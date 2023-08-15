#pragma once

#include "Matrix.cuh"

class Loss
{
public:
	Loss()
	{	}

	void Calculate()
	{
		// Find loss
		Forward();
		m_data_loss = m_negative_log_confidencies->Mean();

		// Find accuracy
		m_predictions_indicies->RowArgmax(m_predictions);
		m_num_predictions_equal_ground_truth->CompareMatrixAndVector(m_predictions_indicies, m_ground_truth);
		m_data_accuracy = m_num_predictions_equal_ground_truth->ColMean();
	}

	float* GetPredictions()
	{
		float* predictions = new float[3];

		cudaMemcpy(predictions, m_predictions->d_matrix, 12, cudaMemcpyDeviceToHost);

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

	Matrix<float>* GetDinputs()
	{
		return m_dinputs;
	}

private:
	virtual void Forward() {}
	virtual void Backward() {}

protected:
	Matrix<float>* m_predictions;
	Matrix<float>* m_ground_truth;
	Matrix<float>* m_one_hot_encoded_ground_truth;

	// Loss
	Matrix<float>* m_negative_log_confidencies;
	float m_data_loss;

	// Accuracy
	Matrix<float>* m_predictions_indicies;
	Matrix<float>* m_num_predictions_equal_ground_truth;
	float m_data_accuracy;

	// Backpropagation
	Matrix<float>* m_dinputs;
};


class CategoricalCrossentropyLoss : public Loss
{
public:
	CategoricalCrossentropyLoss() : Loss()
	{
		// Initialize empty matricies, which will be filled in SetInputs functions
		m_ground_truth = new Matrix<float>;
		m_negative_log_confidencies = new Matrix<float>;
		m_predictions_indicies = new Matrix<float>;
		m_num_predictions_equal_ground_truth = new Matrix<float>;
		m_one_hot_encoded_ground_truth = new Matrix<float>;

		m_clipped_predictions = new Matrix<float>;
		m_correct_confidencies = new Matrix<float>;
		m_dinputs = new Matrix<float>;
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

	void SetInputs(Matrix<float>* predictions, Matrix<float>* ground_truth)
	{
		if (m_ground_truth->Cleared() == false)
		{
			//if (m_predictions != predictions)
			//	delete m_predictions;
			//if (m_ground_truth != ground_truth)
			//	delete m_ground_truth;
			m_negative_log_confidencies->Clear();
			m_predictions_indicies->Clear();
			m_num_predictions_equal_ground_truth->Clear();
			m_one_hot_encoded_ground_truth->Clear();

			m_clipped_predictions->Clear();
			m_correct_confidencies->Clear();
			m_dinputs->Clear();
		}

		if (m_predictions != predictions)
			m_predictions = predictions;

		if (m_ground_truth != ground_truth)
			m_ground_truth = ground_truth;

		m_negative_log_confidencies->InitMatrix(1, m_ground_truth->GetRow());
		m_predictions_indicies->InitMatrix(m_ground_truth->GetRow(), 1);
		m_num_predictions_equal_ground_truth->InitMatrix(m_ground_truth->GetRow(), 1);

		if (m_ground_truth->GetCol() == 1)
			m_one_hot_encoded_ground_truth->InitMatrix(m_ground_truth->GetRow(), m_predictions->GetRow());

		m_clipped_predictions->InitMatrix(m_predictions->GetCol(), m_predictions->GetRow());
		m_correct_confidencies->InitMatrix(1, m_ground_truth->GetRow());

		m_dinputs->InitMatrix(m_predictions->GetCol(), m_predictions->GetRow());
	}

private:
	Matrix<float>* m_clipped_predictions;
	Matrix<float>* m_correct_confidencies;

	void Forward()
	{
		m_clipped_predictions->Clip(m_predictions, 1e-7, 1 - 1e-7);
		m_correct_confidencies->GetValuesAccordingToMatrices(m_clipped_predictions, m_ground_truth);
		m_negative_log_confidencies->NegativeLog(m_correct_confidencies);
	}

	//void ForwardClip()
	//{
	//	m_clipped_predictions->Clip(m_predictions, 1e-7, 1 - 1e-7);
	//	m_correct_confidencies->GetValuesAccordingToMatrices(m_clipped_predictions, m_ground_truth);
	//}
};