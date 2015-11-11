#include <sstream>

#include "Instance.hpp"
#include "Dataset.hpp"

string Instance::toString(const DatasetMetadata* metadata, bool labelOnly) const {
    if (labelOnly) {
        return metadata->classVariable->convertInternalToValue(classLabel);
    } else {
        stringstream ss;
        for (int j = 0; j < metadata->numOfFeatures; ++j) {
            string val = metadata->featureList[j]->convertInternalToValue(featureVector[j]);
            if (metadata->featureList[j]->getType() == "numeric")
                ss << val << ",";
            else
                ss << "'" << val << "',";
        }
        ss << "'" << metadata->classVariable->convertInternalToValue(classLabel) << "'";
        return ss.str();
    }
}