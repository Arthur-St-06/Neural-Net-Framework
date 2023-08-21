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
	Layer(DenseLayer<T>* dense_layer)
	{
		m_dense_layer = dense_layer;
	}

	Layer(ActivationFunction<T>* activation_function)
	{
		m_activation_function = activation_function;
	}

	Layer(SoftmaxCategoricalCrossentropy<T>* softmax_categorical_crossentropy)
	{
		m_softmax_categorical_crossentropy = softmax_categorical_crossentropy;
	}

	DenseLayer<T>* GetDenseLayer()
	{
		return m_dense_layer;
	}

	ActivationFunction<T>* GetActivationFunction()
	{
		return m_activation_function;
	}

	SoftmaxCategoricalCrossentropy<T>* GetSoftmaxCategoricalCrossentropy()
	{
		return m_softmax_categorical_crossentropy;
	}

private:
	DenseLayer<T>* m_dense_layer;
	ActivationFunction<T>* m_activation_function;
	SoftmaxCategoricalCrossentropy<T>* m_softmax_categorical_crossentropy;
};