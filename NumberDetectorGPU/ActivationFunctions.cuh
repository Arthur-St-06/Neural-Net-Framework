#pragma once

#include "Matrix.cuh"

enum class ACTIVATION_TYPE
{
	Relu,
	Sigmoid,
	Softmax,
	Unknown
};

template <class T>
class ActivationFunction
{
public:
	ActivationFunction(Matrix<T>* inputs, ACTIVATION_TYPE activation_type)
		: m_inputs(inputs)
		, m_activation_type(activation_type)
		, m_inputs_row(inputs->GetRow())
		, m_inputs_column(inputs->GetCol())
	{
		m_outputs = new Matrix<T>(m_inputs->GetCol(), m_inputs->GetRow());

		m_dinputs = new Matrix<T>(m_inputs_column, m_inputs_row);

		if (activation_type == ACTIVATION_TYPE::Softmax) {
			m_matrix_row_max = new Matrix<T>(m_inputs->GetCol(), 1);
			m_matrix_row_sum = new Matrix<T>(m_inputs->GetCol(), 1);

			m_single_output = new Matrix<T>(1, m_outputs->GetRow());
			m_single_output_transposed = new Matrix<T>(m_outputs->GetRow(), 1);
			m_single_dvalues = new Matrix<T>(m_outputs->GetRow(), 1);
			m_sample_wise_gradient = new Matrix<T>(m_outputs->GetRow(), 1);
			m_eyed_output = new Matrix<T>(m_outputs->GetRow(), m_outputs->GetRow());
			m_jacobian_matrix = new Matrix<T>(m_outputs->GetRow(), m_outputs->GetRow());
		}
	}

	void Forward()
	{
		if (m_activation_type == ACTIVATION_TYPE::Relu) {
			ForwardRelu();
		}
		else if (m_activation_type == ACTIVATION_TYPE::Sigmoid) {
			ForwardSigmoid();
		}
		else if (m_activation_type == ACTIVATION_TYPE::Softmax) {
			ForwardSoftmax();
		}
	}

	void Backward(Matrix<T>* dvalues)
	{
		if (m_activation_type == ACTIVATION_TYPE::Relu) {
			BackwardReLu(dvalues);
		}
		else if (m_activation_type == ACTIVATION_TYPE::Sigmoid) {
			BackwardSigmoid(dvalues);
		}
		else if (m_activation_type == ACTIVATION_TYPE::Softmax) {
			BackwardSoftmax(dvalues);
		}
	}

	void ForwardRelu()
	{
		m_outputs->Max(0, m_inputs);
	}

	void ForwardSigmoid()
	{

	}

	void ForwardSoftmax()
	{
		// Get unnormalized probabilites by doing exp * (inputs - max of inputs in every row)
		m_matrix_row_max->RowMax(m_inputs);
		m_outputs->SubstractMatrixFromRowValues(m_inputs, m_matrix_row_max);
		m_outputs->Exp(m_outputs);

		// Normalize probabilities by doing outputs / sum of each output row
		m_matrix_row_sum->RowSum(m_outputs);
		m_outputs->DivideMatrixByRow(m_outputs, m_matrix_row_sum);
	}

	void BackwardReLu(Matrix<T>* dvalues)
	{
		m_dinputs->SetMatrix(dvalues);
		m_dinputs->SetZeroIfMatrixValueIsNegative(m_inputs);
	}

	void BackwardSigmoid(Matrix<T>* dvalues)
	{

	}


	// End softmax backward pass when book will talk about it in more details
	void BackwardSoftmax(Matrix<T>* dvalues)
	{
		for (size_t i = 0; i < m_outputs->GetCol(); i++)
		{
			m_single_output->SetRowMatrixToRow(m_outputs, i, 0);
			m_single_output_transposed->SetTransposedMatrix(m_single_output);
			// Multiply every 2 values of matricies
			m_jacobian_matrix->Dot(m_single_output_transposed, m_single_output);

			m_eyed_output->EyeVector(m_single_output);
			m_jacobian_matrix->SubstractMatricies(m_eyed_output, m_jacobian_matrix);

			m_single_dvalues->SetRowMatrixToColumn(dvalues, i, 0);
			m_sample_wise_gradient->Dot(m_jacobian_matrix, m_single_dvalues);
			m_dinputs->SetColMatrixToRow(m_sample_wise_gradient, 0, i);
		}
	}

	void SetOutputs(Matrix<T>* matrix)
	{
		m_outputs = matrix;
	}

	Matrix<T>* GetOutputs()
	{
		return m_outputs;
	}

	Matrix<T>* GetDinputs()
	{
		return m_dinputs;
	}

private:
	Matrix<T>* m_inputs;
	Matrix<T>* m_outputs;

	size_t m_inputs_row;
	size_t m_inputs_column;

	Matrix<T>* m_matrix_row_max;
	Matrix<T>* m_matrix_row_sum;

	Matrix<T>* m_single_output;
	Matrix<T>* m_single_output_transposed;
	Matrix<T>* m_single_dvalues;
	Matrix<T>* m_sample_wise_gradient;
	Matrix<T>* m_eyed_output;
	Matrix<T>* m_jacobian_matrix;
	Matrix<T>* m_dinputs;

	ACTIVATION_TYPE m_activation_type;
};