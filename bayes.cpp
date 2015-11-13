#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>

#include "BayesNet.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "usage: ./bayes train-set-file test-set-file mode:n|t [size-of-train-set] [debug-output:f|t]" << endl;
    } else {
        string trainSetFile = argv[1];
        string testSetFile = argv[2];
        bool treeAugmented = argv[3][0] == 't' ? true : false;
        int sizeOfTrainSet = argc >= 5 ? atoi(argv[4]) : 0;
        bool debugOutput = argc >= 6 ? (argv[5][0] == 't' ? true : false) : false;
        
        shared_ptr<Dataset> dataset(Dataset::loadDataset(trainSetFile, testSetFile));
        const DatasetMetadata* metadata = dataset->getMetadata();
        
        vector<Instance*> trainSet(dataset->getTrainSet().begin(), dataset->getTrainSet().end());
        if (sizeOfTrainSet > 0 && sizeOfTrainSet < trainSet.size()) {
            unsigned int seed = (unsigned int)chrono::system_clock::now().time_since_epoch().count();
            shuffle (trainSet.begin(), trainSet.end(), default_random_engine(seed));
            trainSet.resize(sizeOfTrainSet);
        }
        
        BayesNet bayesNet(metadata, trainSet, treeAugmented);
        
        if (debugOutput) {
            cout << bayesNet.getMutualInfoTable() << endl;
            cout << bayesNet.getMaximalSpanningTree() << endl;
            cout << bayesNet.getProbabilityTables() << endl;
        }
        
        cout << bayesNet.getBayesNet() << endl;
        
        const vector<Instance*>& testSet = dataset->getTestSet();
        int correctCount = 0;
        cout << "<Predictions for Test-set Instances>" << endl;
        cout << "Predicted class" << DELIMITER << "Actual class" << DELIMITER << "Posterior probability" << endl;
        cout.setf(ios::fixed, ios::floatfield);
        cout.precision(PRECISION);
        for (int i = 0; i < testSet.size(); ++i) {
            Instance* inst = testSet[i];
            double prob = 0.0;
            string predicted = bayesNet.predict(inst, &prob);
            string actual = inst->toString(metadata, true);
            
            if (predicted == actual)
                correctCount++;
            
            cout << predicted << DELIMITER << actual << DELIMITER << prob << endl;
        }
        cout << correctCount << " out of " << testSet.size() << " test instances were correctly classified" << endl;
    }
}