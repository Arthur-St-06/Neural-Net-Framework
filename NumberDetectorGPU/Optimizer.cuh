#pragma once

#include "DenseLayer.cuh"

// Stochastic Gradient Descent
template <class T>
class SGD
{
public:
	SGD(float learning_rate = 1.0f)
		: m_learning_rate(learning_rate)
	{	}

	void SetInputs(DenseLayer<T>* layer)
	{
		m_layer = layer;

		m_weights_increment_matrix = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_biases_increment_matrix = new Matrix<T>(1, m_layer->GetBiases()->GetRow());
	}

	void UpdateParams()
	{
		m_weights_increment_matrix->MultByValue(m_layer->GetDweights(), m_learning_rate);
		m_weights_increment_matrix->SubstractValueFromMatrix(m_weights_increment_matrix, 0);
		m_layer->GetWeights()->AddMatrix(m_weights_increment_matrix);

		m_biases_increment_matrix->MultByValue(m_layer->GetDbiases(), m_learning_rate);
		m_biases_increment_matrix->SubstractValueFromMatrix(m_biases_increment_matrix, 0);
		m_layer->GetBiases()->AddMatrix(m_biases_increment_matrix);
	}

private:
	DenseLayer<T>* m_layer;

	Matrix<T>* m_weights_increment_matrix;
	Matrix<T>* m_biases_increment_matrix;

	float m_learning_rate;
};

template <class T>
class RMSprop
{
public:
	RMSprop(float learning_rate = 0.001f, float decay = 0.0f, float epsilon = 1e-7f, float rho = 0.9)
		: m_learning_rate(learning_rate)
		, m_iteration(0)
		, m_decay(decay)
		, m_epsilon(epsilon)
		, m_rho(rho)
	{	}

	void SetInputs(DenseLayer<T>* layer)
	{
		m_layer = layer;

		m_weight_cache = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_cache = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_rho_times_weight_cache = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_rho_times_bias_cache = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_dweights_squared = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_dbiases_squared = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_one_minue_rho_times_dweights_squared = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minue_rho_times_dbiases_squared = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_dweights = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_dbiases = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_square_rooted_weight_cache_plus_epsilon = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_square_rooted_bias_cache_plus_epsilon = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon = new Matrix<T>(1, m_layer->GetBiases()->GetRow());
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
	DenseLayer<T>* m_layer;

	Matrix<T>* m_weight_cache;
	Matrix<T>* m_bias_cache;

	Matrix<T>* m_rho_times_weight_cache;
	Matrix<T>* m_rho_times_bias_cache;

	Matrix<T>* m_dweights_squared;
	Matrix<T>* m_dbiases_squared;

	Matrix<T>* m_one_minue_rho_times_dweights_squared;
	Matrix<T>* m_one_minue_rho_times_dbiases_squared;

	Matrix<T>* m_learning_rate_times_dweights;
	Matrix<T>* m_learning_rate_times_dbiases;

	Matrix<T>* m_square_rooted_weight_cache;
	Matrix<T>* m_square_rooted_bias_cache;

	Matrix<T>* m_square_rooted_weight_cache_plus_epsilon;
	Matrix<T>* m_square_rooted_bias_cache_plus_epsilon;

	Matrix<T>* m_learning_rate_times_dweights_divided_by_square_rooted_weight_cache_plus_epsilon;
	Matrix<T>* m_learning_rate_times_dbiases_divided_by_square_rooted_bias_cache_plus_epsilon;

	float m_learning_rate;
	float m_decayed_learning_rate;
	float m_iteration;
	float m_decay;
	float m_epsilon;
	float m_rho;
};

template <class T>
class Adam
{
public:
	Adam(float learning_rate = 0.02f, float decay = 5e-7f, float epsilon = 1e-7f, float beta_1 = 0.9f, float beta_2 = 0.999f)
		: m_learning_rate(learning_rate)
		, m_iteration(0)
		, m_decay(decay)
		, m_epsilon(epsilon)
		, m_beta1(beta_1)
		, m_beta2(beta_2)
	{	}
	void SetInputs(DenseLayer<T>* layer)
	{
		m_layer = layer;

		m_weight_momentum = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_momentum = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_weight_momentum_corrected = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_momentum_corrected = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_weight_cache = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_bias_cache = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_one_minus_beta1_times_dweights = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minus_beta1_times_dbiases = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_one_minus_beta2_times_dweight_squared = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_one_minus_beta2_times_dbias_squared = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_weight_momentums_corrected = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_bias_momentums_corrected = new Matrix<T>(1, m_layer->GetBiases()->GetRow());

		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon = new Matrix<T>(m_layer->GetWeights()->GetCol(), m_layer->GetWeights()->GetRow());
		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon = new Matrix<T>(1, m_layer->GetBiases()->GetRow());
	}

	void UpdateParams()
	{
		// Decayed learning rate
		m_decayed_learning_rate = m_learning_rate * (1.0f / (1.0f + m_decay * m_iteration));

		// Update momentums
		// 
		// Weights

		m_weight_momentum->MultByValue(m_weight_momentum, m_beta1);
		m_one_minus_beta1_times_dweights->MultByValue(m_layer->GetDweights(), 1 - m_beta1);

		m_weight_momentum->AddMatricies(m_weight_momentum, m_one_minus_beta1_times_dweights);
		// Biases
		m_bias_momentum->MultByValue(m_bias_momentum, m_beta1);
		m_one_minus_beta1_times_dbiases->MultByValue(m_layer->GetDbiases(), 1 - m_beta1);

		m_bias_momentum->AddMatricies(m_bias_momentum, m_one_minus_beta1_times_dbiases);

		// Update corrected momentums
		// 
		// Weights
		m_weight_momentum_corrected->DivideMatrixByValue(m_weight_momentum, 1 - std::pow(m_beta1, m_iteration + 1));

		// Biases
		m_bias_momentum_corrected->DivideMatrixByValue(m_bias_momentum, 1 - std::pow(m_beta1, m_iteration + 1));

		// Update cache
		// 
		// Weights
		m_weight_cache->MultByValue(m_weight_cache, m_beta2);
		m_one_minus_beta2_times_dweight_squared->PowerMatrix(m_layer->GetDweights(), 2);
		m_one_minus_beta2_times_dweight_squared->MultByValue(m_one_minus_beta2_times_dweight_squared, 1 - m_beta2);

		m_weight_cache->AddMatricies(m_weight_cache, m_one_minus_beta2_times_dweight_squared);

		// Biases
		m_bias_cache->MultByValue(m_bias_cache, m_beta2);
		m_one_minus_beta2_times_dbias_squared->PowerMatrix(m_layer->GetDbiases(), 2);
		m_one_minus_beta2_times_dbias_squared->MultByValue(m_one_minus_beta2_times_dbias_squared, 1 - m_beta2);

		m_bias_cache->AddMatricies(m_bias_cache, m_one_minus_beta2_times_dbias_squared);

		// Update cache corrected
		//
		// Weights
		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon->DivideMatrixByValue(m_weight_cache, 1 - std::pow(m_beta2, m_iteration + 1));

		// Biases

		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon->DivideMatrixByValue(m_bias_cache, 1 - std::pow(m_beta2, m_iteration + 1));

		// Update parameters
		// 
		// Weights
		m_learning_rate_times_weight_momentums_corrected->MultByValue(m_weight_momentum_corrected, m_decayed_learning_rate);
		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon->SqrtMatrix(m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon);

		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon->AddValue(m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon, m_epsilon);
		m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon->DivideMatrices(m_learning_rate_times_weight_momentums_corrected, m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon);

		m_layer->GetWeights()->SubstractMatricies(m_layer->GetWeights(), m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon);

		// Biases
		m_learning_rate_times_bias_momentums_corrected->MultByValue(m_bias_momentum_corrected, m_decayed_learning_rate);
		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon->SqrtMatrix(m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon);

		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon->AddValue(m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon, m_epsilon);
		m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon->DivideMatrices(m_learning_rate_times_bias_momentums_corrected, m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon);

		m_layer->GetBiases()->SubstractMatricies(m_layer->GetBiases(), m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon);

		// Update iteration
		m_iteration++;
	}

private:
	DenseLayer<T>* m_layer;

	Matrix<T>* m_weight_momentum;
	Matrix<T>* m_bias_momentum;

	Matrix<T>* m_weight_momentum_corrected;
	Matrix<T>* m_bias_momentum_corrected;

	Matrix<T>* m_weight_cache;
	Matrix<T>* m_bias_cache;

	Matrix<T>* m_one_minus_beta1_times_dweights;
	Matrix<T>* m_one_minus_beta1_times_dbiases;

	Matrix<T>* m_one_minus_beta2_times_dweight_squared;
	Matrix<T>* m_one_minus_beta2_times_dbias_squared;

	Matrix<T>* m_learning_rate_times_weight_momentums_corrected;
	Matrix<T>* m_learning_rate_times_bias_momentums_corrected;

	Matrix<T>* m_learning_rate_times_weight_momentums_corrected_divided_by_square_rooted_weight_cache_corrected_plus_epsilon;
	Matrix<T>* m_learning_rate_times_bias_momentums_corrected_divided_by_square_rooted_bias_cache_corrected_plus_epsilon;

	float m_learning_rate;
	float m_decayed_learning_rate;
	float m_iteration;
	float m_decay;
	float m_epsilon;
	float m_beta1;
	float m_beta2;
};

template <class T>
class Optimizer
{
public:
	Optimizer(SGD<T>* sgd)
	{
		m_sgd = sgd;
	}

	Optimizer(RMSprop<T>* rmsprop)
	{
		m_rmsprop = rmsprop;
	}

	Optimizer(Adam<T>* adam)
	{
		m_adam = adam;
	}

	SGD<T>* GetSGD()
	{
		return m_sgd;
	}

	RMSprop<T>* GetRMSprop()
	{
		return m_rmsprop;
	}

	Adam<T>* GetAdam()
	{
		return m_adam;
	}

private:
	SGD<T>* m_sgd;
	RMSprop<T>* m_rmsprop;
	Adam<T>* m_adam;
};