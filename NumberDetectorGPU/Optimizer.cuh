#pragma once

#include "DenseLayer.cuh"

// Stochastic Gradient Descent
class SGD
{
public:
	SGD(DenseLayer<float>* layer, float learning_rate = 1.0f, float decay = 0.0f, float momentum = 0.0f)
		: m_layer(layer)
		, m_learning_rate(learning_rate)
		, m_iteration(0)
		, m_decay(decay)
		, m_momenutm(momentum)
	{
		m_weights_increment_matrix = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_biases_increment_matrix = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		if (m_momenutm != 0)
		{
			//m_weight_momentums = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
			//m_bias_momentums = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

			m_weight_updates = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
			m_bias_updates = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

			m_weight_momentums = m_weight_updates;
			m_bias_momentums = m_bias_updates;
		}
	}

	void UpdateParams()
	{
		// Decayed learning rate
		m_decayed_learning_rate = m_learning_rate * (1.0f / (1.0f + m_decay * m_iteration));

		if (m_momenutm != 0)
		{
			m_weight_updates->MultByValue(m_weight_momentums, m_momenutm);
			m_weights_increment_matrix->MultByValue(m_layer->GetDweights(), m_decayed_learning_rate);
			m_weight_updates->SubstractMatricies(m_weight_updates, m_weights_increment_matrix);

			m_weight_momentums = m_weight_updates;

			m_bias_updates->MultByValue(m_bias_momentums, m_momenutm);
			m_biases_increment_matrix->MultByValue(m_layer->GetDbiases(), m_decayed_learning_rate);
			m_bias_updates->SubstractMatricies(m_bias_updates, m_biases_increment_matrix);

			m_bias_momentums = m_bias_updates;
		}

		// Update parameters
		m_layer->GetWeights()->AddMatrix(m_weight_updates);
		m_layer->GetBiases()->AddMatrix(m_bias_momentums);

		// Update iteration
		m_iteration++;
	}

private:
	DenseLayer<float>* m_layer;

	Matrix<float>* m_weights_increment_matrix;
	Matrix<float>* m_biases_increment_matrix;

	Matrix<float>* m_weight_momentums;
	Matrix<float>* m_bias_momentums;

	Matrix<float>* m_weight_updates;
	Matrix<float>* m_bias_updates;

	float m_learning_rate;
	float m_decayed_learning_rate;
	float m_iteration;
	float m_decay;
	float m_momenutm;
};

class RMSprop
{
public:
	RMSprop(DenseLayer<float>* layer, float learning_rate = 0.001f, float decay = 0.0f, float epsilon = 1e-7f, float rho = 0.9f)
		: m_layer(layer)
		, m_learning_rate(learning_rate)
		, m_iteration(0)
		, m_decay(decay)
		, m_epsilon(epsilon)
		, m_rho(rho)
	{
		m_weight_cache = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_cache = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_rho_times_weight_cache = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_rho_times_bias_cache = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_dweights_squared = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_dbiases_squared = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_one_minue_rho_times_dweights_squared = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minue_rho_times_dbiases_squared = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_dweights = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_dbiases = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache_plus_epsilon = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache_plus_epsilon = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon = new Matrix<float>(1, m_layer->GetBiases()->GetRow());
	}

	void UpdateParams()
	{
		// Decayed learning rate
		m_decayed_learning_rate = m_learning_rate * (1.0f / (1.0f + m_decay * m_iteration));

		// Update cache with squared current gradients
		// 
		// Weights
		m_rho_times_weight_cache->MultByValue(m_weight_cache, m_rho);
		m_dweights_squared->PowerMatrix(m_layer->GetDweights(), 2);
		m_one_minue_rho_times_dweights_squared->MultByValue(m_dweights_squared, 1 - m_rho);

		m_weight_cache->AddMatricies(m_rho_times_weight_cache, m_one_minue_rho_times_dweights_squared);

		// Biases
		m_rho_times_bias_cache->MultByValue(m_bias_cache, m_rho);
		m_dbiases_squared->PowerMatrix(m_layer->GetDbiases(), 2);
		m_one_minue_rho_times_dbiases_squared->MultByValue(m_dbiases_squared, 1 - m_rho);

		m_bias_cache->AddMatricies(m_rho_times_bias_cache, m_one_minue_rho_times_dbiases_squared);

		// Update parameters
		// 
		// Weights
		m_learning_rate_times_dweights->MultByValue(m_layer->GetDweights(), m_decayed_learning_rate);
		m_square_rooted_weight_cache->SqrtMatrix(m_weight_cache);
		m_square_rooted_weight_cache_plus_epsilon->AddValue(m_square_rooted_weight_cache, m_epsilon);
		m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon->DivideMatrices(m_learning_rate_times_dweights, m_square_rooted_weight_cache_plus_epsilon);

		m_layer->GetWeights()->SubstractMatricies(m_layer->GetWeights(), m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon);

		// Biases
		m_learning_rate_times_dbiases->MultByValue(m_layer->GetDbiases(), m_decayed_learning_rate);
		m_square_rooted_bias_cache->SqrtMatrix(m_bias_cache);
		m_square_rooted_bias_cache_plus_epsilon->AddValue(m_square_rooted_bias_cache, m_epsilon);
		m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon->DivideMatrices(m_learning_rate_times_dbiases, m_square_rooted_bias_cache_plus_epsilon);

		m_layer->GetBiases()->SubstractMatricies(m_layer->GetBiases(), m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon);

		// Update iteration
		m_iteration++;
	}

private:
	DenseLayer<float>* m_layer;

	Matrix<float>* m_weight_cache;
	Matrix<float>* m_bias_cache;

	Matrix<float>* m_rho_times_weight_cache;
	Matrix<float>* m_rho_times_bias_cache;

	Matrix<float>* m_dweights_squared;
	Matrix<float>* m_dbiases_squared;

	Matrix<float>* m_one_minue_rho_times_dweights_squared;
	Matrix<float>* m_one_minue_rho_times_dbiases_squared;

	Matrix<float>* m_learning_rate_times_dweights;
	Matrix<float>* m_learning_rate_times_dbiases;

	Matrix<float>* m_square_rooted_weight_cache;
	Matrix<float>* m_square_rooted_bias_cache;

	Matrix<float>* m_square_rooted_weight_cache_plus_epsilon;
	Matrix<float>* m_square_rooted_bias_cache_plus_epsilon;

	Matrix<float>* m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon;
	Matrix<float>* m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon;

	float m_learning_rate;
	float m_decayed_learning_rate;
	float m_iteration;
	float m_decay;
	float m_epsilon;
	float m_rho;
};

class Adam
{
public:
	Adam()
	{	}
	void SetInputs(DenseLayer<float>* layer, float learning_rate = 0.02f, float decay = 5e-7, float epsilon = 1e-7f, float beta_1 = 0.9f, float beta_2 = 0.999f)
	{
		m_layer = layer;
		m_learning_rate = learning_rate;
		m_iteration = 0;
		m_decay = decay;
		m_epsilon = epsilon;
		m_beta1 = beta_1;
		m_beta2 = beta_2;

		m_weight_momentums = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_momentums = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_weight_cache = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_cache = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_beta1_times_weight_momentums = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_beta1_times_bias_momentums = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_one_minus_beta1_times_dweights = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minus_beta1_times_dbiases = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_weight_momentums_corrected = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_momentums_corrected = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_beta2_times_weight_cache = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_beta2_times_bias_cache = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_dweights_squared = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_dbiases_squared = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_one_minus_beta2_times_dweight_squared = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minus_beta2_times_dbias_squared = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_weight_cache_corrected = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_cache_corrected = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_weight_momentums_corrected = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_bias_momentums_corrected = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache_corrected = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache_corrected = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache_corrected_plus_epsilon = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache_corrected_plus_epsilon = new Matrix<float>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon = new Matrix<float>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon = new Matrix<float>(1, m_layer->GetBiases()->GetRow());
	}

	void UpdateParams()
	{
		// Decayed learning rate
		m_decayed_learning_rate = m_learning_rate * (1.0f / (1.0f + m_decay * m_iteration));

		// Update momentums
		// 
		// Weights

		m_beta1_times_weight_momentums->MultByValue(m_weight_momentums, m_beta1);

		m_one_minus_beta1_times_dweights->MultByValue(m_layer->GetDweights(), 1 - m_beta1);
		m_weight_momentums->AddMatricies(m_beta1_times_weight_momentums, m_one_minus_beta1_times_dweights);

		// Biases
		m_beta1_times_bias_momentums->MultByValue(m_bias_momentums, m_beta1);
		m_one_minus_beta1_times_dbiases->MultByValue(m_layer->GetDbiases(), 1 - m_beta1);

		m_bias_momentums->AddMatricies(m_beta1_times_bias_momentums, m_one_minus_beta1_times_dbiases);

		// Update corrected momentums
		// 
		// Weights
		m_weight_momentums_corrected->DivideMatrixByValue(m_weight_momentums, 1 - std::pow(m_beta1, m_iteration + 1));

		// Biases
		m_bias_momentums_corrected->DivideMatrixByValue(m_bias_momentums, 1 - std::pow(m_beta1, m_iteration + 1));

		// Update cache
		// 
		// Weights
		m_beta2_times_weight_cache->MultByValue(m_weight_cache, m_beta2);
		m_dweights_squared->PowerMatrix(m_layer->GetDweights(), 2);

		m_one_minus_beta2_times_dweight_squared->MultByValue(m_dweights_squared, 1 - m_beta2);

		m_weight_cache->AddMatricies(m_beta2_times_weight_cache, m_one_minus_beta2_times_dweight_squared);

		// Biases
		m_beta2_times_bias_cache->MultByValue(m_bias_cache, m_beta2);
		m_dbiases_squared->PowerMatrix(m_layer->GetDbiases(), 2);
		m_one_minus_beta2_times_dbias_squared->MultByValue(m_dbiases_squared, 1 - m_beta2);

		m_bias_cache->AddMatricies(m_beta2_times_bias_cache, m_one_minus_beta2_times_dbias_squared);

		// Update cache corrected
		//
		// Weights
		m_weight_cache_corrected->DivideMatrixByValue(m_weight_cache, 1 - std::pow(m_beta2, m_iteration + 1));

		// Biases

		m_bias_cache_corrected->DivideMatrixByValue(m_bias_cache, 1 - std::pow(m_beta2, m_iteration + 1));

		// Update parameters
		// 
		// Weights
		m_learning_rate_times_weight_momentums_corrected->MultByValue(m_weight_momentums_corrected, m_decayed_learning_rate);
		m_square_rooted_weight_cache_corrected->SqrtMatrix(m_weight_cache_corrected);

		m_square_rooted_weight_cache_corrected_plus_epsilon->AddValue(m_square_rooted_weight_cache_corrected, m_epsilon);
		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon->DivideMatrices(m_learning_rate_times_weight_momentums_corrected, m_square_rooted_weight_cache_corrected_plus_epsilon);

		m_layer->GetWeights()->SubstractMatricies(m_layer->GetWeights(), m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon);

		// Biases
		m_learning_rate_times_bias_momentums_corrected->MultByValue(m_bias_momentums_corrected, m_decayed_learning_rate);
		m_square_rooted_bias_cache_corrected->SqrtMatrix(m_bias_cache_corrected);
		m_square_rooted_bias_cache_corrected_plus_epsilon->AddValue(m_square_rooted_bias_cache_corrected, m_epsilon);
		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon->DivideMatrices(m_learning_rate_times_bias_momentums_corrected, m_square_rooted_bias_cache_corrected_plus_epsilon);

		m_layer->GetBiases()->SubstractMatricies(m_layer->GetBiases(), m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon);

		// Update iteration
		m_iteration++;
	}

private:
	DenseLayer<float>* m_layer;

	Matrix<float>* m_weight_momentums;
	Matrix<float>* m_bias_momentums;

	Matrix<float>* m_weight_cache;
	Matrix<float>* m_bias_cache;

	Matrix<float>* m_beta1_times_weight_momentums;
	Matrix<float>* m_beta1_times_bias_momentums;

	Matrix<float>* m_one_minus_beta1_times_dweights;
	Matrix<float>* m_one_minus_beta1_times_dbiases;

	Matrix<float>* m_weight_momentums_corrected;
	Matrix<float>* m_bias_momentums_corrected;

	Matrix<float>* m_beta2_times_weight_cache;
	Matrix<float>* m_beta2_times_bias_cache;

	Matrix<float>* m_dweights_squared;
	Matrix<float>* m_dbiases_squared;

	Matrix<float>* m_one_minus_beta2_times_dweight_squared;
	Matrix<float>* m_one_minus_beta2_times_dbias_squared;

	Matrix<float>* m_weight_cache_corrected;
	Matrix<float>* m_bias_cache_corrected;

	Matrix<float>* m_learning_rate_times_weight_momentums_corrected;
	Matrix<float>* m_learning_rate_times_bias_momentums_corrected;

	Matrix<float>* m_square_rooted_weight_cache_corrected;
	Matrix<float>* m_square_rooted_bias_cache_corrected;

	Matrix<float>* m_square_rooted_weight_cache_corrected_plus_epsilon;
	Matrix<float>* m_square_rooted_bias_cache_corrected_plus_epsilon;

	Matrix<float>* m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon;
	Matrix<float>* m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon;

	float m_learning_rate;
	float m_decayed_learning_rate;
	float m_iteration;
	float m_decay;
	float m_epsilon;
	float m_beta1;
	float m_beta2;
};

class Optimizer
{
public:
	Optimizer(SGD* sgd)
	{
		m_sgd = sgd;
	}

	Optimizer(RMSprop* rmsprop)
	{
		m_rmsprop = rmsprop;
	}

	Optimizer(Adam* adam)
	{
		m_adam = adam;
	}

	SGD* GetSGD()
	{
		return m_sgd;
	}

	RMSprop* GetRMSprop()
	{
		return m_rmsprop;
	}

	Adam* GetAdam()
	{
		return m_adam;
	}

private:
	SGD* m_sgd;
	RMSprop* m_rmsprop;
	Adam* m_adam;
};