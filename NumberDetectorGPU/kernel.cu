
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "Matrix.cuh"
#include "DenseLayer.cuh"
#include "ActivationFunctions.cuh"
#include "Loss.cuh"
#include "SoftmaxCategoricalCrossentropy.cuh"
#include "Optimizer.cuh"
#include "Data.cuh"

int main()
{
	// Initialize time variables to avoid speed latency of functions
	auto begin = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	begin = std::chrono::high_resolution_clock::now();

	// Stop measuring time and calculate the elapsed time
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	begin = std::chrono::high_resolution_clock::now();

	Data<float> data;

	DenseLayer<float> dense1(2, 64, data.GetTrainingDataInputs(), INIT_TYPE::Xavier_Normal, 0, 0, 5e-4, 5e-4);
	ActivationFunction<float> Activation1(dense1.GetOutputs(), ACTIVATION_TYPE::Relu);

	DenseLayer<float> dense2(64, 64, Activation1.GetOutputs(), INIT_TYPE::Xavier_Normal, 0, 0, 5e-4, 5e-4);
	ActivationFunction<float> Activation2(dense2.GetOutputs(), ACTIVATION_TYPE::Relu);

	DenseLayer<float> dense3(64, 3, Activation2.GetOutputs(), INIT_TYPE::Xavier_Normal);
	SoftmaxCategoricalCrossentropy<float> SoftmaxLoss(dense3.GetOutputs(), data.GetTrainingDataOutputs());

	float learning_rate = 0.02f;
	float decay = 5e-7;
	float epsilon = 1e-7f;
	float beta1 = 0.9;
	float beta2 = 0.999f;

	//float momentum = 0.8f;
	//float rho = 0.9999f;

	Adam optimizer1(&dense1, learning_rate, decay, epsilon, beta1, beta2);
	Adam optimizer2(&dense2, learning_rate, decay, epsilon, beta1, beta2);
	Adam optimizer3(&dense3, learning_rate, decay, epsilon, beta1, beta2);

	//RMSprop optimizer1(&dense1, learning_rate, decay, epsilon, rho);
	//RMSprop optimizer2(&dense2, learning_rate, decay, epsilon, rho);

	//SGD optimizer1(&dense1, learning_rate, decay, momentum);
	//SGD optimizer2(&dense2, learning_rate, decay, momentum);

	float reg_loss;

	for (size_t epoch = 0; epoch < 1001; epoch++)
	{
		dense1.Forward();		
		Activation1.Forward();

		dense2.Forward();
		Activation2.Forward();

		dense3.Forward();
		SoftmaxLoss.Forward();

		if (epoch % 10 == 0)
		{
			reg_loss = 0;
			reg_loss += dense1.RegularizationLoss();
			reg_loss += dense2.RegularizationLoss();

			std::cout << "Epoch: " << epoch;
			std::cout << ", loss: " << SoftmaxLoss.GetLoss()->GetLoss();
			std::cout << ", reg loss: " << reg_loss;
			std::cout << ", accuracy: " << SoftmaxLoss.GetLoss()->GetAccuracy() << std::endl;
		}

		SoftmaxLoss.Backward();
		dense3.Backward(SoftmaxLoss.GetDinputs());
		Activation2.Backward(dense3.GetDinputs());
		dense2.Backward(Activation2.GetDinputs());
		Activation1.Backward(dense2.GetDinputs());
		dense1.Backward(Activation1.GetDinputs());
		
		optimizer1.UpdateParams();
		optimizer2.UpdateParams();
		optimizer3.UpdateParams();
	}

	dense1.SetInputs(data.GetValidatingDataInputs());
	SoftmaxLoss.SetGroundTruth(data.GetTrainingDataOutputs());

	dense1.Forward();
	Activation1.Forward();

	dense2.Forward();
	Activation2.Forward();

	dense3.Forward();
	SoftmaxLoss.Forward();

	std::cout << "Loss: " << SoftmaxLoss.GetLoss()->GetLoss();
	std::cout << ", accuracy: " << SoftmaxLoss.GetLoss()->GetAccuracy() << std::endl;

	// Stop measuring time and calculate the elapsed time
	end = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	printf("Time measured GPU: %.3f seconds.\n", elapsed.count() * 1e-9);

	return 0;
}