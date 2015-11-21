#ifndef BayesNet_hpp
#define BayesNet_hpp

#include "Dataset.hpp"

const char DELIMITER = ' ';
const int PRECISION = 16;

struct CPT {
protected:
    int self;
    vector<int> parents;
    
    CPT(int self, const vector<int>& parents) : self(self), parents(parents) {};
    
public:
    virtual ~CPT() {};
    virtual void buildTable(const vector<Instance*>& instances) = 0;
    virtual double computeCondProb(const Instance* instance) const = 0;
    virtual string toString() const = 0;
};

struct CPT0 : public CPT {
private:
    vector<double> table;
    
public:
    CPT0(const DatasetMetadata* metadata, int self, const vector<int>& parents) : CPT(self, parents) {
        Feature* selfFeature = metadata->classVariable;
        table.resize(selfFeature->getRange());
    }
    
    virtual void buildTable(const vector<Instance*>& instances);
    virtual double computeCondProb(const Instance* instance) const;
    virtual string toString() const;
};

struct CPT1 : public CPT {
private:
    vector<vector<double> > table;
    
public:
    CPT1(const DatasetMetadata* metadata, int self, const vector<int>& parents) : CPT(self, parents) {
        Feature* selfFeature = metadata->featureList[self];
        Feature* parentFeature = metadata->classVariable;
        
        table.resize(selfFeature->getRange());
        for (int i = 0; i < table.size(); ++i)
            table[i].resize(parentFeature->getRange());
    }
    
    virtual void buildTable(const vector<Instance*>& instances);
    virtual double computeCondProb(const Instance* instance) const;
    virtual string toString() const;
};

struct CPT2 : public CPT {
private:
    vector<vector<vector<double> > > table;
    
public:
    CPT2(const DatasetMetadata* metadata, int self, const vector<int>& parents) : CPT(self, parents) {
        Feature* selfFeature =  metadata->featureList[self];
        Feature* parentFeature0 = metadata->featureList[parents[0]];
        Feature* parentFeature1 = metadata->classVariable;
        
        table.resize(selfFeature->getRange());
        for (int i = 0; i < table.size(); ++i) {
            table[i].resize(parentFeature0->getRange());
            for (int j = 0; j < table[i].size(); ++j)
                table[i][j].resize(parentFeature1->getRange());
        }
    }
    
    virtual void buildTable(const vector<Instance*>& instances);
    virtual double computeCondProb(const Instance* instance) const;
    virtual string toString() const;
};

class BayesNet {
private:
    const DatasetMetadata* metadata;
    const vector<Instance*>& instances;
    bool treeAugmented;
    
    vector<vector<double> > mutualInfoTable;
    vector<pair<int, int> > maximalSpanningTree;
    vector<vector<int> > bayesNet;
    vector<CPT*> probabilityTables;
    
    double computeMutualInfo(int featureIdxI, int featureIdxJ) const;
    CPT* computeCPT(int self, const vector<int>& parents) const;
    
    void createMutualInfoTable();
    void createMaximalSpanningTree();
    void createBayesNet();
    void createProbabilityTables();
    
public:
    BayesNet(const DatasetMetadata* metadata, const vector<Instance*>& instances, bool treeAugmented);
    
    ~BayesNet() {
        for (int i = 0; i < probabilityTables.size(); ++i)
            if (probabilityTables[i])
                delete probabilityTables[i];
    }
    
    const DatasetMetadata* getMetadata() const {
        return metadata;
    }
    
    string getMutualInfoTable() const;
    string getMaximalSpanningTree() const;
    string getBayesNet() const;
    string getProbabilityTables() const;
    
    string predict(const Instance* instance, double* probability = 0) const;
};

#endif /* BayesNet_hpp */
