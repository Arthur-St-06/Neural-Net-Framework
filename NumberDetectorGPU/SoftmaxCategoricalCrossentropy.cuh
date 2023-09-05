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
		// Create instances of ActivationFunction and CategoricalCrossentropyLoss
		m_softmax = new ActivationFunction<T>(ACTIVATION_TYPE::Softmax);
		m_loss = new CategoricalCrossentropyLoss<T>();

		m_ground_truth = new Matrix<T>();
		m_softmax_inputs = new Matrix<T>();
		m_dinputs = new Matrix<T>();
	}

	// Set the input matrices for Softmax and ground truth
	void SetInputs(Matrix<T>* softmax_inputs, Matrix<T>* ground_truth)
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
			m_ground_truth = ground_truth;

		m_softmax->SetInputs(m_softmax_inputs);
		m_loss->SetInputs(m_softmax->GetOutputs(), m_ground_truth);

		if(m_dinputs->Cleared() == false)
			m_dinputs->Clear();
		m_dinputs->InitMatrix(m_softmax->GetOutputs()->GetCol(), m_softmax->GetOutputs()->GetRow());
	}

	// Perform the forward pass of Softmax and calculate the loss
	void Forward(Matrix<T>* ground_truth)
	{
		m_ground_truth = ground_truth;

		m_softmax->Forward();
		m_loss->Calculate(m_ground_truth);
	}

	// Perform the backward pass
	void Backward()
	{
		m_dinputs->SetMatrix(m_softmax->GetOutputs());
		m_dinputs->SubstractMatrixFromValueAtMatrixIdx(m_ground_truth, 1);
		m_dinputs->DivideMatrixByValue(m_dinputs, m_dinputs->GetCol());
	}

	ActivationFunction<T>* GetSoftmax()
	{
		return m_softmax;
	}

	CategoricalCrossentropyLoss<T>* GetLoss()
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
	CategoricalCrossentropyLoss<T>* m_loss;

	Matrix<T>* m_dinputs;
};