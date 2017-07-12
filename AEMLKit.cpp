//
// Created by Wang Jinli on 2017/6/14.
//

#include "AEMLKit.h"
#include <shark/Models/Softmax.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;
using namespace std;

FeedForwardNet::FeedForwardNet(size_t numInputs, size_t numHiddenLayers, size_t numClasses, size_t *hiddenLayerSizes) {
    FFNetStructure s = { numInputs, numHiddenLayers, numClasses, hiddenLayerSizes };
    structure = s;
}

void FeedForwardNet::train(TrainingData *dataset, size_t count, unsigned int epochs) {
    trainingData = convertData(dataset, count);
    CrossEntropy loss;
    ErrorFunction error(trainingData, &ffnet, &loss);
    IRpropPlus optimizer;
    error.init();
    optimizer.init(error);
    for (unsigned int i = 0; i < epochs; i++) {
        optimizer.step(error);
    }
    ffnet.setParameterVector(optimizer.solution().point);
}

double FeedForwardNet::validate(ValidatingData *dataset, size_t count) {
    ClassificationDataset validatingData = convertData(dataset, count);
    ZeroOneLoss<unsigned int, RealVector> loss01;
    Data<RealVector> prediction = ffnet(validatingData.inputs());
    return loss01.eval(validatingData.labels(), prediction);
}

PredictResult FeedForwardNet::predict(double *input) {
    RealVector inputVec;
    for (size_t i = 0; i < structure.numInputs; i++) {
        inputVec.push_back(input[i]);
    }
    Softmax probability(structure.numClasses);
    RealVector probabilities = (ffnet >> probability)(inputVec);
    unsigned int label = (unsigned int)distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));
    double *probs = new double[structure.numClasses];
    for (size_t i = 0; i < structure.numClasses; i++) {
        probs[i] = probabilities[i];
    }
    PredictResult result = { label, probs };
    return result;
}

void FeedForwardNet::initFFNet() {
    FFNet<LogisticNeuron, LinearNeuron> network;
    vector<size_t> layers;
    layers.push_back(structure.numInputs);
    for (size_t i = 0; i < structure.numHiddenLayers; i++) {
        layers.push_back(structure.hiddenLayerSizes[i]);
    }
    layers.push_back(structure.numClasses);
    network.setStructure(layers, FFNetStructures::Normal, true);
    initRandomUniform(network, -0.1, 0.1);
    ffnet = network;
}

shark::ClassificationDataset FeedForwardNet::convertData(TrainingData *dataset, size_t count) {
    vector<RealVector> inputs;
    vector<unsigned int> labels;
    for (size_t i = 0; i < count; i++) {
        RealVector input;
        for (size_t j = 0; j < structure.numInputs; j++) {
            input.push_back(dataset[i].input[j]);
        }
        inputs.push_back(input);
        labels.push_back(dataset[i].label);
    }
    return createLabeledDataFromRange(inputs, labels);
}
