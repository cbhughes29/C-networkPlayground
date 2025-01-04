#include "neuralNetworkCore.h"

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

// Layer

Layer::Layer(int numberOfNeurons, int numberOfInputs, Neuron inputNeuron) : numberOfNeurons(numberOfNeurons), numberOfInputs(numberOfInputs), neuron(inputNeuron), weights(numberOfNeurons, std::vector<double>(numberOfInputs, 0.0)), biases(numberOfNeurons)  {

    if (numberOfInputs <= 0) {
        throw std::invalid_argument("Layer: numberOfInputs must be greater than zero");
    }
    if (numberOfNeurons <= 0) {
        throw std::invalid_argument("Layer: numberOfNeurons must be greater than zero");
    }

    // Initialize random weights
    for (int i = 0; i < numberOfNeurons; i++) {
        for (int j = 0; j < numberOfInputs; j++) {
            double randomDoubleForWeight = uniform_distribution(generator);
            weights[i][j] = randomDoubleForWeight;
        }
        biases[i] = uniform_distribution(generator);
    }
}


std::vector<double> Layer::computeLayerAction(std::vector<double>& inputs) {
    std::vector<double> output(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; i++) {
        double sum = 0;
        for (int j = 0; j < numberOfInputs; j++) {
            sum += inputs[j] * weights[i][j];
        }
        sum += biases[i];
        output[i] = neuron.activate(sum);
    }
    return output;
}

// Network

NeuralNetwork::NeuralNetwork(int numInputs, int numLayers, std::vector<int> numNeuronsPerLayer, std::vector<std::function<double(double)>>& activationFunctionsPerLayer) : inputSize(numInputs), numberOfLayers(numLayers) {
    if (numInputs <= 0) {
        throw std::invalid_argument("NeuralNetwork: numberOfInputs must be greater than zero");
    }
    if (numLayers <= 0) {
        throw std::invalid_argument("NeuralNetwork: numberOfLayers must be greater than zero");
    }
    if (numNeuronsPerLayer.size() != numLayers) {
        throw std::invalid_argument("NeuralNetwork: numNeuronsPerLayer must be equal to numLayers");
    }
    if (activationFunctionsPerLayer.size() != numLayers) {
        throw std::invalid_argument("NeuralNetwork: activationFunctionsPerLayer must be equal to numLayers");
    }

    // Create layers
    int currentNumInputs = numInputs;
    neuronsPerLayer = numNeuronsPerLayer;

    for (int i = 0; i < numLayers; i++) {
        Neuron neuronForCurrentLayer(activationFunctionsPerLayer[i]);
        int numberOfNeuronsForCurrentLayer = numNeuronsPerLayer[i];
        Layer currentLayer(numberOfNeuronsForCurrentLayer, currentNumInputs, neuronForCurrentLayer);
        layers.push_back(currentLayer);
        currentNumInputs = numberOfNeuronsForCurrentLayer;
    }
    outputSize = currentNumInputs;
}

std::vector<double> NeuralNetwork::forwardPropagate(std::vector<double>& inputs) {
    if (inputs.size() != inputSize) {
        throw std::invalid_argument("NeuralNetwork: number of inputs must be equal to inputSize");
    }

    std::vector<double> output(outputSize);

    std::vector<double> currentInput = inputs;
    for (int i = 0; i < numberOfLayers; i++) {
        currentInput = layers[i].computeLayerAction(currentInput);
    }
    return currentInput;
}



