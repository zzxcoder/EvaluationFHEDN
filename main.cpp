#include "detector.h"
#include "evaluation.h"
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    const std::string& model_file = "/home/zzx/work/EvaluationFHEDN/models/FHEDN_512x512/deploy.prototxt";
    const std::string& weights_file = "/home/zzx/work/EvaluationFHEDN/models/FHEDN_512x512/weights/VGG_WIDERFACE_FHEDN_512x512.caffemodel";
    const std::string& mean_file = "";
    const std::string& mean_value = "104,117,123";

    Detector detector(model_file, weights_file, mean_file, mean_value);
    detectImage(&detector);
    evaluateAFWList(&detector);
    evaluatePascalList(&detector);
    evaluateFDDBList(&detector);
    evaluateWIDERFACEList(&detector);

    return 0;
}
