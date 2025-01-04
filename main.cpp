#include <neuralNetworkCore.h>

int main() {

    std::vector<int> numNeuronsPerLayer;
    for (int i = 0; i < 3; i++) {
        numNeuronsPerLayer.push_back(3);
    }
    std::vector<std::function<double(double)>> functionVec;
    for (int i = 0; i < 3; i++) {
        functionVec.push_back(relu);
    }

    NeuralNetwork firstNetwork(2, 3, numNeuronsPerLayer, functionVec);
    std::vector<double> inputVec;
    inputVec.push_back(1);
    inputVec.push_back(2);
    std::vector<double> output = firstNetwork.forwardPropagate(inputVec);

    return 0;
}
