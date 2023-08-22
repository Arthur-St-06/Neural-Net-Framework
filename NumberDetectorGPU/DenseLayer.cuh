#pragma once

#include "Matrix.cuh"

template <class T>
class DenseLayer
{
public:
	DenseLayer(size_t n_inputs, size_t n_outputs, INIT_TYPE init_type = INIT_TYPE::Xavier_Normal,
		float weight_regularizer_l1 = 0.0f, float bias_regularizer_l1 = 0.0f, float weight_regularizer_l2 = 5e-4f, float bias_regularizer_l2 = 5e-4f)
		: m_column(n_inputs)
		, m_row(n_outputs)
		, m_init_type(init_type)
		, m_weight_regularizer_l1(weight_regularizer_l1)
		, m_bias_regularizer_l1(bias_regularizer_l1)
		, m_weight_regularizer_l2(weight_regularizer_l2)
		, m_bias_regularizer_l2(bias_regularizer_l2)
	{
		// Initialize empty matricies, which will be filled in SetInputs functions
		m_inputs = new Matrix<T>;
		m_outputs = new Matrix<T>;
		m_dinputs = new Matrix<T>;

		// m_weights is automatically transposed as it has random initialization type
		m_weights = new Matrix<T>(m_column, m_row, m_init_type);
		m_biases = new Matrix<T>(1, m_row);

		m_dweights = new Matrix<T>(m_column, m_row);
		m_dbiases = new Matrix<T>(1, m_row);

		// Initialize regularizars derivatives
		if (m_weight_regularizer_l1 > 0)
			m_weights_dl1 = new Matrix<T>(m_column, m_row); 

		if (m_bias_regularizer_l1 > 0) 
			m_biases_dl1 = new Matrix<T>(1, m_row);

		if (m_weight_regularizer_l2 > 0) 
			m_weights_dl2 = new Matrix<T>(m_column, m_row);

		if (m_bias_regularizer_l2 > 0)
			m_biases_dl2 = new Matrix<T>(1, m_row);
	}

	void SetInputs(Matrix<T>* inputs)
	{
		m_inputs_row = inputs->GetRow();
		m_inputs_column = inputs->GetCol();

		if (m_inputs->Cleared() == false)
		{
			m_outputs->Clear();
			m_dinputs->Clear();
		}
		
		if(m_inputs != inputs)
			m_inputs = inputs;
		m_outputs->InitMatrix(m_inputs->GetCol(), m_row);
		m_dinputs->InitMatrix(m_inputs_column, m_inputs_row);
	}

	void Forward()
	{
		m_outputs->Dot(m_inputs, m_weights);
		//cudaMemcpy(result, m_outputs->d_matrix, 768000, cudaMemcpyDeviceToHost);
		m_outputs->AddSingleRow(m_biases);
	}

	void Backward(Matrix<T>* dvalues)
	{
		// Gradients on parameters

		//m_transposed_inputs->SetTransposedMatrix(m_inputs);

		m_dweights->Dot(m_inputs, dvalues, "T");

		m_dbiases->ColSum(dvalues);

		// Gradients on regularization
		// 
		// L1 on weights

		// L1 on biases

		// L2 on weights
		if (m_weight_regularizer_l2 > 0)
		{
			m_weights_dl2->MultByValue(m_weights, 2 * m_weight_regularizer_l2);

			m_dweights->AddMatrix(m_weights_dl2);
		}

		// L2 on biases
		if (m_bias_regularizer_l2 > 0)
		{
			m_biases_dl2->MultByValue(m_biases, 2 * m_bias_regularizer_l2);

			m_dbiases->AddMatrix(m_biases_dl2);
		}

		// Gradient on inputs
		m_dinputs->Dot(dvalues, m_weights, "T2");
	}

	float RegularizationLoss()
	{
		m_regularization_loss = 0.0f;
		
		if (m_weight_regularizer_l1 > 0)
		{
			m_weights_dl1->Abs(m_weights);
			m_regularization_loss += m_weight_regularizer_l1 * m_weights_dl1->Sum(m_weights_dl1);
		}
		if (m_bias_regularizer_l1 > 0)
		{
			m_biases_dl1->Abs(m_biases);
			m_regularization_loss += m_bias_regularizer_l1 * m_biases_dl1->Sum(m_biases_dl1);
		}
		if (m_weight_regularizer_l2 > 0)
		{
			m_weights_dl2->PowerMatrix(m_weights, 2);
			m_regularization_loss = m_weight_regularizer_l2 * m_weights_dl2->Sum(m_weights_dl2);
		}
		if (m_bias_regularizer_l2 > 0)
		{
			m_biases_dl2->PowerMatrix(m_biases, 2);
			m_regularization_loss += m_bias_regularizer_l2 * m_biases_dl2->Sum(m_biases_dl2);
		}	

		return m_regularization_loss;
	}

	Matrix<T>* GetWeights()
	{
		return m_weights;
	}

	Matrix<T>* GetBiases()
	{
		return m_biases;
	}

	Matrix<T>* GetOutputs()
	{
		return m_outputs;
	}

	Matrix<T>* GetDinputs()
	{
		return m_dinputs;
	}

	Matrix<T>* GetDweights()
	{
		return m_dweights;
	}

	Matrix<T>* GetDbiases()
	{
		return m_dbiases;
	}

	float GetWeightRegularizerL1()
	{
		return m_weight_regularizer_l1;
	}

	float GetBiasRegularizerL1()
	{
		return m_bias_regularizer_l1;
	}

	float GetWeightRegularizerL2()
	{
		return m_weight_regularizer_l2;
	}

	float GetBiasRegularizerL2()
	{
		return m_bias_regularizer_l2;
	}	

private:
	size_t m_row;
	size_t m_column;

	size_t m_inputs_row;
	size_t m_inputs_column;

	Matrix<T>* m_inputs;
	Matrix<T>* m_weights;
	Matrix<T>* m_biases;
	Matrix<T>* m_outputs;

	// Backpropagation
	Matrix<T>* m_dinputs;
	Matrix<T>* m_dweights;
	Matrix<T>* m_dbiases;

	// Regularization
	Matrix<T>* m_weights_dl1;
	Matrix<T>* m_biases_dl1;
	Matrix<T>* m_weights_dl2;
	Matrix<T>* m_biases_dl2;

	float m_regularization_loss;

	// Regularization strength
	// l1
	float m_weight_regularizer_l1;
	float m_bias_regularizer_l1;

	// l2
	float m_weight_regularizer_l2;
	float m_bias_regularizer_l2;

	INIT_TYPE m_init_type;
};