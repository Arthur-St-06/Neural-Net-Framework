#pragma once

#include "Matrix.cuh"

class Loss
{
public:
	Loss(Matrix<float>* predictions, Matrix<float>* ground_truth)
		: m_predictions(predictions)
		, m_ground_truth(ground_truth)
	{
		m_negative_log_confidencies = new Matrix<float>(1, m_ground_truth->GetRow());
		m_predictions_indicies = new Matrix<float>(m_ground_truth->GetRow(), 1);
		m_num_predictions_equal_ground_truth = new Matrix<float>(m_ground_truth->GetRow(), 1);

		if (m_ground_truth->GetCol() == 1)
		{
			m_one_hot_encoded_ground_truth = new Matrix<float>(m_ground_truth->GetRow(), m_predictions->GetRow());
		}
	}

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

	//void CalculateForward()
	//{
	//	// Find loss
	//	ForwardClip();
	//}
	//
	//void CalculateDataLoss()
	//{
	//	m_data_loss = m_negative_log_confidencies->Mean();
	//}
	//
	//void CalculateAccuracy()
	//{
	//	// Find accuracy
	//	m_predictions_indicies->RowArgmax(m_predictions);
	//	m_num_predictions_equal_ground_truth->CompareMatrixAndVector(m_predictions_indicies, m_ground_truth);
	//	m_data_accuracy = m_num_predictions_equal_ground_truth->ColMean();
	//}

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
	CategoricalCrossentropyLoss(Matrix<float>* predictions, Matrix<float>* ground_truth) : Loss(predictions, ground_truth)

	{
		m_clipped_predictions = new Matrix<float>(m_predictions->GetCol(), m_predictions->GetRow());
		m_correct_confidencies = new Matrix<float>(1, m_ground_truth->GetRow());

		m_dinputs = new Matrix<float>(m_predictions->GetCol(), m_predictions->GetRow());
	}

	void Backward()
	{
		if (m_ground_truth->GetCol() == 1)
		{
			m_one_hot_encoded_ground_truth->OneHotEncode(m_ground_truth);
		}

		// Calculate gradient
		m_dinputs->DivideMatrices(m_one_hot_encoded_ground_truth, m_predictions);

		m_dinputs->SubstractValueFromMatrix(m_dinputs, 0);
		// Normalize gradient for optimizer as it sums all will sum all of the samples to one
		m_dinputs->DivideMatrixByValue(m_dinputs, m_dinputs->GetCol());
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