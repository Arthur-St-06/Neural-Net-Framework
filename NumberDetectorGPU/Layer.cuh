#include <stdio.h>
#include <iostream>
#include<vector>

#include "Matrix.cuh"
#include "DenseLayer.cuh"
#include "ActivationFunctions.cuh"
#include "Loss.cuh"
#include "SoftmaxCategoricalCrossentropy.cuh"
#include "Optimizer.cuh"
#include "Data.cuh"

template <class T>
class Layer
{
public:
	Layer(DenseLayer<float>* dense_layer)
	{
		m_dense_layer = dense_layer;
	}

	Layer(ActivationFunction<float>* activation_function)
	{
		m_activation_function = activation_function;
	}

	Layer(SoftmaxCategoricalCrossentropy<float>* softmax_categorical_crossentropy)
	{
		m_softmax_categorical_crossentropy = softmax_categorical_crossentropy;
	}

	DenseLayer<float>* GetDenseLayer()
	{
		return m_dense_layer;
	}

	ActivationFunction<float>* GetActivationFunction()
	{
		return m_activation_function;
	}

	SoftmaxCategoricalCrossentropy<float>* GetSoftmaxCategoricalCrossentropy()
	{
		return m_softmax_categorical_crossentropy;
	}

private:
	DenseLayer<float>* m_dense_layer;
	ActivationFunction<float>* m_activation_function;
	SoftmaxCategoricalCrossentropy<float>* m_softmax_categorical_crossentropy;
};