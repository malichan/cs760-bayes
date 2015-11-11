#ifndef Instance_hpp
#define Instance_hpp

#include <vector>
#include <string>

using namespace std;

struct DatasetMetadata;

struct Instance {
public:
    vector<double> featureVector;
    double classLabel;
    
    Instance(int numOfFeatures) : featureVector(numOfFeatures, -1), classLabel(-1) {}
    string toString(const DatasetMetadata* metadata, bool labelOnly = false) const;
};

#endif /* Instance_hpp */
