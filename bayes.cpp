#include <iostream>
#include <memory>

#include "BayesNet.hpp"

const bool DEBUG_MODE = true;

int main(int argc, char* argv[]) {
    string trainSetFile = "vote_train.arff";
    string testSetFile = "vote_test.arff";
    
    shared_ptr<Dataset> dataset(Dataset::loadDataset(trainSetFile, testSetFile));
    const DatasetMetadata* metadata = dataset->getMetadata();
    
    vector<Instance*> trainSet(dataset->getTrainSet().begin(), dataset->getTrainSet().end());
    vector<Instance*> testSet(dataset->getTestSet().begin(), dataset->getTestSet().end());
    
    BayesNet bayesNet(metadata, trainSet);
    
    if (DEBUG_MODE) {
        cout << bayesNet.getMutualInfoTable() << endl;
        cout << bayesNet.getMaximalSpanningTree() << endl;
        cout << bayesNet.getProbabilityTables() << endl;
    }
    
    cout << bayesNet.getBayesNet() << endl;
    
    cout << "<Predictions for the Test-set Instances>" << endl;
    cout << "Predicted class\tActual class\tPosterior probability" << endl;
    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(6);
    int correctCount = 0;
    for (int i = 0; i < testSet.size(); ++i) {
        Instance* inst = testSet[i];
        double prob = 0.0;
        string predicted = bayesNet.predict(inst, &prob);
        string actual = inst->toString(metadata, true);
        
        if (predicted == actual)
            correctCount++;
        
        cout << predicted << "\t" << actual << "\t" << prob << endl;
    }
    cout << correctCount << " out of " << testSet.size() << " test instances were correctly classified" << endl;
}