#pragma once

#include "Matrix.cuh"
#include "ActivationFunctions.cuh"
#include "Loss.cuh"

template <class T>
class SoftmaxCategoricalCrossentropy
{
public:
	SoftmaxCategoricalCrossentropy(Matrix<T>* softmax_inputs, Matrix<T>* ground_truth)
		: m_softmax_inputs(softmax_inputs)
		, m_ground_truth(ground_truth)
	{
		m_softmax = new ActivationFunction<float>(m_softmax_inputs, ACTIVATION_TYPE::Softmax);
		m_loss = new CategoricalCrossentropyLoss(m_softmax->GetOutputs(), m_ground_truth);

		m_dinputs = new Matrix<T>(m_softmax->GetOutputs()->GetCol(), m_softmax->GetOutputs()->GetRow());
	}

	void Forward()
	{
		m_softmax->Forward();
		m_loss->Calculate();
	}

	//void ForwardSoft()
	//{
	//	m_softmax->Forward();
	//}
	//
	//void Calcloss()
	//{
	//	m_loss->CalculateForward();
	//}

	void Backward()
	{
		m_dinputs->SetMatrix(m_softmax->GetOutputs());
		m_dinputs->SubstractMatrixFromValueAtMatrixIdx(m_dinputs, m_ground_truth, 1);
		m_dinputs->DivideMatrixByValue(m_dinputs, m_dinputs->GetCol());
	}

	void SetGroundTruth(Matrix<float>* ground_truth) { m_ground_truth = ground_truth; }

	ActivationFunction<T>* GetSoftmax()
	{
		return m_softmax;
	}

	CategoricalCrossentropyLoss* GetLoss()
	{
		return m_loss;
	}

	Matrix<T>* GetDinputs()
	{
		return m_dinputs;
	}

private:
	Matrix<T>* m_softmax_inputs;
	Matrix<T>* m_ground_truth;

	ActivationFunction<T>* m_softmax;
	CategoricalCrossentropyLoss* m_loss;

	Matrix<T>* m_dinputs;
};