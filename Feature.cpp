#include <sstream>
#include <cmath>

#include "Feature.hpp"

string NumericFeature::toString() const {
    stringstream ss;
    ss << "@attribute '" << getName() << "' numeric";
    return ss.str();
}

string NominalFeature::toString() const {
    stringstream ss;
    ss << "@attribute '" << getName() << "' {'" << idxToName[0] << "'";
    for (int j = 1; j < getRange(); ++j)
        ss << ",'" << idxToName[j] << "'";
    ss << "}";
    return ss.str();
}

double NumericFeature::convertValueToInternal(const string& str) const {
    return stod(str);
}

double NominalFeature::convertValueToInternal(const string& str) const {
    unordered_map<string, int>::const_iterator it = nameToIdx.find(str);
    if (it != nameToIdx.end())
        return it->second;
    else
        return -1;
    
}

string NumericFeature::convertInternalToValue(double val) const {
    stringstream ss;
    ss.setf(ios::fixed, ios::floatfield);
    ss.precision(6);
    ss << val;
    return ss.str();
}

string NominalFeature::convertInternalToValue(double val) const {
    return idxToName[(int)round(val)];
}