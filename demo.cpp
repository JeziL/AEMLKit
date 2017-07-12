//
// Created by Wang Jinli on 2017/6/14.
//

#include <iostream>
#include <fstream>
#include "AEMLKit.h"

using namespace std;

double **readCSV(const char *filename, int row, int column) {
    double **data = 0;
    data = new double *[row];
    ifstream ifs;
    ifs.clear();
    ifs.open(filename);
    string s; char c;
    for (int i = 0; i < row; i++) {
        data[i] = new double[column];
        getline(ifs, s);
        stringstream stream(s);
        int j = 0;
        while(1) {
            stream >> data[i][j];
            stream >> c;
            j++;
            if (!stream) { break; }
        }
    }
    ifs.close();
    return data;
}

int main() {
    // Import training data.
    int trainingCount = 120;
    double **t = readCSV("data_train.csv", trainingCount, 3);
    TrainingData *trainingData = new TrainingData[trainingCount];
    for (int i = 0; i < trainingCount; i++) {
        // 0~39:   Label 0
        // 40~79:  Label 1
        // 80~119: Label 2
        trainingData[i] = { t[i], (unsigned int)(i / 40) };
    }

    // Define the network structure.
    size_t hiddenSizes[] = {10};
    FeedForwardNet network(3, 1, 3, hiddenSizes);
    network.initFFNet();

    // Train.
    unsigned int epochs = 200;
    network.train(trainingData, (size_t)trainingCount, epochs);

    // Import validating data.
    int validatingCount = 30;
    double **v = readCSV("data_validate.csv", validatingCount, 3);
    ValidatingData *validatingData = new ValidatingData[validatingCount];
    for (int i = 0; i < validatingCount; i++) {
        // 0~9:   Label 0
        // 10~19: Label 1
        // 20~29: Label 2
        validatingData[i] = { v[i], (unsigned int)(i / 10) };
    }

    // Print validating result.
    cout << "Error rate on validation set:\t" << network.validate(validatingData, 30) << endl;

    // Import test data.
    int testCount = 30;
    double **testData = readCSV("data_test.csv", testCount, 3);

    // Predict test data.
    for (int i = 0; i < testCount; i++) {
        PredictResult r = network.predict(testData[i]);
        cout << "[" << r.predictedLabel << "]\t" << r.probabilities[0] << " " << r.probabilities[1] << " " << r.probabilities[2] << endl;
        delete[] r.probabilities;
    }

    delete[] t;
    delete[] trainingData;
    delete[] v;
    delete[] validatingData;
    delete[] testData;
    return 0;
}
