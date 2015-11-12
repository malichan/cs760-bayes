#include <sstream>
#include <set>

#include "BayesNet.hpp"

void CPT0::buildTable(const vector<Instance*>& instances) {
    int rangeY = (int)table.size();
    
    vector<int> YOccurance;
    YOccurance.resize(rangeY);
    
    for (int i = 0; i < instances.size(); ++i) {
        Instance* inst = instances[i];
        int valY = (int)round(inst->classLabel);
        YOccurance[valY]++;
    }
    
    int total = (int)instances.size();
    for (int valY = 0; valY < rangeY; ++valY) {
        table[valY] = (YOccurance[valY] + 1.0) / (total + rangeY);
    }
}

double CPT0::computeCondProb(const Instance* instance) const {
    int valSelf = (int)round(instance->classLabel);
    return table[valSelf];
}

string CPT0::toString() const {
    stringstream ss;
    ss << "CPT of attribute " << self << endl;
    
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    for (int i = 0; i < table.size(); ++i)
        ss << "Pr(" << self << " = " << i << ") = " << table[i] << endl;
    
    return ss.str();
}

void CPT1::buildTable(const vector<Instance*>& instances) {
    int rangeX = (int)table.size();
    int rangeY = (int)table[0].size();
    
    vector<int> YOccurance;
    YOccurance.resize(rangeY);
    vector<vector<int> > XYOccurance;
    XYOccurance.resize(rangeX);
    for (int valX = 0; valX < rangeX; ++valX)
        XYOccurance[valX].resize(rangeY);
    
    for (int i = 0; i < instances.size(); ++i) {
        Instance* inst = instances[i];
        int valX = (int)round(inst->featureVector[self]);
        int valY = (int)round(inst->classLabel);
        YOccurance[valY]++;
        XYOccurance[valX][valY]++;
    }
    
    for (int valX = 0; valX < rangeX; ++valX)
        for (int valY = 0; valY < rangeY; ++valY)
            table[valX][valY] = (XYOccurance[valX][valY] + 1.0) / (YOccurance[valY] + rangeX);
}

double CPT1::computeCondProb(const Instance* instance) const {
    int valSelf = (int)round(instance->featureVector[self]);
    int valParent = (int)round(instance->classLabel);
    return table[valSelf][valParent];
}

string CPT1::toString() const {
    stringstream ss;
    ss << "CPT of attribute " << self << endl;
    
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    for (int j = 0; j < table[0].size(); ++j)
        for (int i = 0; i < table.size(); ++i)
            ss << "Pr(" << self << " = " << i << " | " << parents[0] << " = " << j << ") = " << table[i][j] << endl;
    
    return ss.str();
}

void CPT2::buildTable(const vector<Instance*>& instances) {
    int rangeX = (int)table.size();
    int rangeZ = (int)table[0].size();
    int rangeY = (int)table[0][0].size();
    
    vector<vector<int> > ZYOccurance;
    ZYOccurance.resize(rangeZ);
    for (int valZ = 0; valZ < rangeZ; ++valZ)
        ZYOccurance[valZ].resize(rangeY);
    vector<vector<vector<int> > > XZYOccurance;
    XZYOccurance.resize(rangeX);
    for (int valX = 0; valX < rangeX; ++valX) {
        XZYOccurance[valX].resize(rangeZ);
        for (int valZ = 0; valZ < rangeZ; ++valZ)
            XZYOccurance[valX][valZ].resize(rangeY);
    }
    
    for (int i = 0; i < instances.size(); ++i) {
        Instance* inst = instances[i];
        int valX = (int)round(inst->featureVector[self]);
        int valZ = (int)round(inst->featureVector[parents[0]]);
        int valY = (int)round(inst->classLabel);
        ZYOccurance[valZ][valY]++;
        XZYOccurance[valX][valZ][valY]++;
    }
    
    for (int valX = 0; valX < rangeX; ++valX)
        for (int valZ = 0; valZ < rangeZ; ++valZ)
            for (int valY = 0; valY < rangeY; ++valY)
                table[valX][valZ][valY] = (XZYOccurance[valX][valZ][valY] + 1.0) / (ZYOccurance[valZ][valY] + rangeX);
}

double CPT2::computeCondProb(const Instance* instance) const {
    int valSelf = (int)round(instance->featureVector[self]);
    int valParent0 = (int)round(instance->featureVector[parents[0]]);
    int valParent1 = (int)round(instance->classLabel);
    return table[valSelf][valParent0][valParent1];
}

string CPT2::toString() const {
    stringstream ss;
    ss << "CPT of attribute " << self << endl;
    
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    for (int k = 0; k < table[0][0].size(); ++k)
        for (int j = 0; j < table[0].size(); ++j)
            for (int i = 0; i < table.size(); ++i)
                ss << "Pr(" << self << " = " << i << " | " << parents[0] << " = " << j << ", " <<
                    parents[1] << " = " << k << ") = " << table[i][j][k] << endl;
    
    return ss.str();
}

BayesNet::BayesNet(const DatasetMetadata* metadata, const vector<Instance*>& instances) : metadata(metadata), instances(instances) {
    createMutualInfoTable();
    createMaximalSpanningTree();
    createBayesNet();
    createProbabilityTables();
}

double BayesNet::computeMutualInfo(int featureIdxI, int featureIdxJ) const {
    Feature* Y = metadata->classVariable;
    Feature* Xi = metadata->featureList[featureIdxI];
    Feature* Xj = metadata->featureList[featureIdxJ];
    
    int rangeY = Y->getRange();
    int rangeXi = Xi->getRange();
    int rangeXj = Xj->getRange();
    
    vector<int> YOccurance;
    YOccurance.resize(rangeY);
    vector<vector<int> > YXiOccurance;
    YXiOccurance.resize(rangeY);
    for (int valY = 0; valY < rangeY; ++valY)
        YXiOccurance[valY].resize(rangeXi);
    vector<vector<int> > YXjOccurance;
    YXjOccurance.resize(rangeY);
    for (int valY = 0; valY < rangeY; ++valY)
        YXjOccurance[valY].resize(rangeXj);
    vector<vector<vector<int> > > YXiXjOccurance;
    YXiXjOccurance.resize(rangeY);
    for (int valY = 0; valY < rangeY; ++valY) {
        YXiXjOccurance[valY].resize(rangeXi);
        for (int valXi = 0; valXi < rangeXi; ++valXi)
            YXiXjOccurance[valY][valXi].resize(rangeXj);
    }
    
    for (int i = 0; i < instances.size(); ++i) {
        Instance* inst = instances[i];
        int valY = (int)round(inst->classLabel);
        int valXi = (int)round(inst->featureVector[featureIdxI]);
        int valXj = (int)round(inst->featureVector[featureIdxJ]);
        YOccurance[valY]++;
        YXiOccurance[valY][valXi]++;
        YXjOccurance[valY][valXj]++;
        YXiXjOccurance[valY][valXi][valXj]++;
    }
    
    int total = (int)instances.size();
    double mutualInfo = 0.0;
    for (int valY = 0; valY < rangeY; ++valY) {
        for (int valXi = 0; valXi < rangeXi; ++valXi) {
            for (int valXj = 0; valXj < rangeXj; ++valXj) {
                double pXiXjY = (YXiXjOccurance[valY][valXi][valXj] + 1.0) /
                    (total + rangeXi * rangeXj * rangeY);
                double pXiXj_Y = (YXiXjOccurance[valY][valXi][valXj] + 1.0) /
                    (YOccurance[valY] + rangeXi * rangeXj);
                double pXi_Y = (YXiOccurance[valY][valXi] + 1.0) /
                    (YOccurance[valY] + rangeXi);
                double pXj_Y = (YXjOccurance[valY][valXj] + 1.0) /
                    (YOccurance[valY] + rangeXj);
                mutualInfo += pXiXjY * log2(pXiXj_Y / (pXi_Y * pXj_Y));
            }
        }
    }
    
    return mutualInfo;
}

void BayesNet::createMutualInfoTable() {
    int numOfFeatures = metadata->numOfFeatures;
    
    mutualInfoTable.resize(numOfFeatures);
    for (int i = 0; i < numOfFeatures; ++i)
        mutualInfoTable[i].resize(numOfFeatures);
    
    
    for (int i = 0; i < numOfFeatures; ++i) {
        mutualInfoTable[i][i] = -1.0;
        for (int j = i + 1; j < numOfFeatures; ++j) {
            double mutualInfo = computeMutualInfo(i, j);
            mutualInfoTable[i][j] = mutualInfo;
            mutualInfoTable[j][i] = mutualInfo;
        }
    }
}

string BayesNet::getMutualInfoTable() const {
    stringstream ss;
    ss << "<Conditional Mutual Information Table>" << endl;
    
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    for (int i = 0; i < metadata->numOfFeatures; ++i) {
        for (int j = 0; j < metadata->numOfFeatures; ++j) {
            if (j != 0) ss << "\t";
            ss << mutualInfoTable[i][j];
        }
        ss << endl;
    }
    
    return ss.str();
}

void BayesNet::createMaximalSpanningTree() {
    int numOfFeatures = metadata->numOfFeatures;

    set<int> nodesInTree;
    set<int> nodesNotInTree;
    nodesInTree.insert(0);
    for (int i = 1; i < numOfFeatures; ++i)
        nodesNotInTree.insert(i);
    
    while (nodesInTree.size() < numOfFeatures) {
        double maxWeight = -1.0;
        int maxI = -1;
        int maxJ = -1;
        for (set<int>::iterator itI = nodesInTree.begin(); itI != nodesInTree.end(); ++itI) {
            int i = *itI;
            for (set<int>::iterator itJ = nodesNotInTree.begin(); itJ != nodesNotInTree.end(); ++itJ) {
                int j = *itJ;
                if (mutualInfoTable[i][j] > maxWeight) {
                    maxWeight = mutualInfoTable[i][j];
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        nodesInTree.insert(maxJ);
        nodesNotInTree.erase(maxJ);
        maximalSpanningTree.push_back(pair<int, int>(maxI, maxJ));
    }
}

string BayesNet::getMaximalSpanningTree() const {
    stringstream ss;
    ss << "<Maximal Spanning Tree>" << endl;
    
    ss << "{";
    for (int i = 0; i < maximalSpanningTree.size(); ++i) {
        if (i != 0) ss << ", ";
        ss << "(" << maximalSpanningTree[i].first << ", " << maximalSpanningTree[i].second << ")";
    }
    ss << "}" << endl;
    
    return ss.str();
}

void BayesNet::createBayesNet() {
    int numOfFeatures = metadata->numOfFeatures;
    
    bayesNet.resize(numOfFeatures);
    
    for (int i = 0; i < maximalSpanningTree.size(); ++i) {
        pair<int, int> edge = maximalSpanningTree[i];
        bayesNet[edge.second].push_back(edge.first);
    }
    
    for (int i = 0; i < numOfFeatures; ++i)
        bayesNet[i].push_back(numOfFeatures);
}

string BayesNet::getBayesNet() const {
    stringstream ss;
    ss << "<Bayes Net Structure>" << endl;
    
    for (int i = 0; i < bayesNet.size(); ++i) {
        ss << metadata->featureList[i]->getName();
        for (int j = 0; j < bayesNet[i].size(); ++j) {
            ss << "\t";
            int featureIdx = bayesNet[i][j];
            if (featureIdx < metadata->numOfFeatures)
                ss << metadata->featureList[featureIdx]->getName();
            else
                ss << metadata->classVariable->getName();
        }
        ss << endl;
    }
    
    return ss.str();
}

CPT* BayesNet::computeCPT(int self, const vector<int>& parents) const {
    CPT* cpt = 0;
    switch (parents.size()) {
        case 0:
            cpt = new CPT0(metadata, self, parents);
            break;
        case 1:
            cpt = new CPT1(metadata, self, parents);
            break;
        case 2:
            cpt = new CPT2(metadata, self, parents);
            break;
        default:
            break;
    }
    if (cpt) cpt->buildTable(instances);
    return cpt;
}

void BayesNet::createProbabilityTables() {
    int numOfFeatures = metadata->numOfFeatures;
    
    probabilityTables.resize(numOfFeatures + 1);
    
    for (int i = 0; i < numOfFeatures; ++i)
        probabilityTables[i] = computeCPT(i, bayesNet[i]);
    probabilityTables[numOfFeatures] = computeCPT(numOfFeatures, vector<int>());
}

string BayesNet::getProbabilityTables() const {
    stringstream ss;
    ss << "<Conditional Probability Tables>" << endl;
    
    for (int i = 0; i < probabilityTables.size(); ++i)
        ss << probabilityTables[i]->toString();
    
    return ss.str();
}

string BayesNet::predict(const Instance* instance, double* probability) const {
    int numOfClasses = metadata->numOfClasses;
    int numOfFeatures = metadata->numOfFeatures;
    
    Instance inst = *instance;
    double probSum = 0.0;
    vector<double> probs(numOfClasses);
    for (int y = 0; y < numOfClasses; ++y) {
        inst.classLabel = y;
        probs[y] = probabilityTables.back()->computeCondProb(&inst);
        for (int x = 0; x < numOfFeatures; ++x) {
            probs[y] *= probabilityTables[x]->computeCondProb(&inst);
        }
        probSum += probs[y];
    }
    
    double maxProb = -1.0;
    int maxClass = -1;
    for (int y = 0; y < numOfClasses; ++y) {
        probs[y] /= probSum;
        if (probs[y] > maxProb) {
            maxProb = probs[y];
            maxClass = y;
        }
    }
    
    if (probability)
        *probability = maxProb;
    return metadata->classVariable->convertInternalToValue(maxClass);
}