#pragma once

#include "Matrix.cuh"
#include "ActivationFunctions.cuh"
#include "Loss.cuh"

template <class T>
class SoftmaxCategoricalCrossentropy
{
public:
	SoftmaxCategoricalCrossentropy()
	{
		m_softmax = new ActivationFunction<float>(ACTIVATION_TYPE::Softmax);
		m_loss = new CategoricalCrossentropyLoss();

		m_ground_truth = new Matrix<T>();
		m_softmax_inputs = new Matrix<T>();

		m_dinputs = new Matrix<T>();
	}

	void SetInputs(Matrix<float>* softmax_inputs, Matrix<float>* ground_truth)
	{
		if (m_softmax_inputs != softmax_inputs)
		{
			if (m_softmax_inputs->Cleared() == false)
			{
				delete m_softmax_inputs;
			}
			m_softmax_inputs = softmax_inputs;
		}

		if (m_ground_truth != ground_truth)
		{
			if (m_ground_truth->Cleared() == false)
			{
				//delete m_ground_truth;
			}
			m_ground_truth = ground_truth;
		}

		m_softmax->SetInputs(m_softmax_inputs);
		m_loss->SetInputs(m_softmax->GetOutputs(), m_ground_truth);

		if(m_dinputs->Cleared() == false)
			m_dinputs->Clear();
		m_dinputs->InitMatrix(m_softmax->GetOutputs()->GetCol(), m_softmax->GetOutputs()->GetRow());
	}

	void Forward()
	{
		m_softmax->Forward();
		m_loss->Calculate();
	}

	void Backward()
	{
		m_dinputs->SetMatrix(m_softmax->GetOutputs());
		m_dinputs->SubstractMatrixFromValueAtMatrixIdx(m_dinputs, m_ground_truth, 1);
		m_dinputs->DivideMatrixByValue(m_dinputs, m_dinputs->GetCol());
	}

	//void SetGroundTruth(Matrix<float>* ground_truth) { m_ground_truth = ground_truth; }

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