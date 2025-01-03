//
// Created by Christian on 1/3/2025.
//
#include "neuralNetworkCore.h"
#include <vector>
#include <functional>
#include <math.h>
#include <valarray>
#include <random>
#include <stdexcept>

double randWeightsLowerBound = 0.0;
double randWeightsUpperBound = 1.0;
std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
std::default_random_engine generator;

// Activation functions

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivatives of activation functions

double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
}

// Neuron, Layer, and Network classes

class Neuron {
private:
    std::function<double(double)> activationFunction;
public:
    Neuron(std::function<double(double)>& activationFunction) {}
    double activate(double x) {
        return activationFunction(x);
    }
};

class Layer {
private:
    int numberOfNeurons;
    int numberOfInputs;
    std::vector<std::vector<double>> weights; // Each row contains weights for one neuron
    std::vector<double> biases;
    Neuron neuron;
public:
    Layer(int numberOfNeurons, int numberOfInputs, Neuron inputNeuron) : numberOfNeurons(numberOfNeurons), numberOfInputs(numberOfInputs), neuron(inputNeuron) {

        if (numberOfInputs <= 0) {
            throw std::invalid_argument("Layer: numberOfInputs must be greater than zero");
        }
        if (numberOfNeurons <= 0) {
            throw std::invalid_argument("Layer: numberOfNeurons must be greater than zero");
        }

        // Initialize random weights
        std::vector<std::vector<double>> weights(numberOfNeurons, std::vector<double>(numberOfInputs, 0.0));
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                double randomDoubleForWeight = uniform_distribution(generator);
                weights[i][j] = randomDoubleForWeight;
            }
            double randomDoubleForBias = uniform_distribution(generator);
            biases[i] = randomDoubleForBias;
        }
    }

    std::vector<double> computeLayerAction(const std::vector<double>& inputs) {
        std::vector<double> output;
        for (int i = 0; i < numberOfNeurons; i++) {
            double sum = 0;
            for (int j = 0; j < numberOfInputs; j++) {
                double inputTimesWeight = inputs[j] * weights[i][j];
                sum += inputTimesWeight;
            }
            sum += biases[i];
            output[i] = neuron.activate(sum);
        }
        return output;
    }

};


class NeuralNetwork {
private:
    int inputSize;
    int numberOfLayers;
    std::vector<int> neuronsPerLayer;
    std::vector<Layer> layers;
public:
    NeuralNetwork(int numInputs, int numLayers, std::vector<int> numNeuronsPerLayer, std::vector<std::function<double(double)>>& activationFunctionsPerLayer) : inputSize(numInputs), numberOfLayers(numLayers) {
        if (numInputs <= 0) {
            throw std::invalid_argument("NeuralNetwork: numberOfInputs must be greater than zero");
        }
        if (numLayers <= 0) {
            throw std::invalid_argument("NeuralNetwork: numberOfLayers must be greater than zero");
        }

        neuronsPerLayer = numNeuronsPerLayer;
        // Create layers

    }

};



//
