#pragma once
#include <math.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <functional>
#include <math.h>
#include <valarray>
#include <random>
#include <stdexcept>

// Base components for standard artifical neural networks

// Activation functions

double relu(double x);
double sigmoid(double x);

// Derivatives of activation functions

double sigmoidDerivative(double x);
double reluDerivative(double x);

// Loss functions

double MSE(std::vector<double>& ypred, std::vector<double>& y);

double MAE(std::vector<double>& ypred, std::vector<double>& y);

// Neuron

class Neuron {
private:
    std::function<double(double)> activationFunction;
public:
    Neuron(std::function<double(double)>& activationFunc) :  activationFunction(activationFunc) {}
    double activate(double x) {
        return activationFunction(x);
    }
};

// Layer

class Layer {
private:
    int numberOfNeurons;
    int numberOfInputs;
    std::vector<std::vector<double>> weights; // Each row contains weights for one neuron
    std::vector<double> biases;
    Neuron neuron;
public:
    Layer(int numberOfNeurons, int numberOfInputs, Neuron inputNeuron);
    std::vector<double> computeLayerAction(std::vector<double>& inputs);
};

// Network

class NeuralNetwork {
private:
    int inputSize;
    int numberOfLayers;
    std::vector<int> neuronsPerLayer;
    std::vector<Layer> layers;
    int outputSize;
public:
    NeuralNetwork(int numInputs, int numLayers, std::vector<int> numNeuronsPerLayer, std::vector<std::function<double(double)>>& activationFunctionsPerLayer);
    std::vector<double> forwardPropagate(std::vector<double>& inputs);
};

