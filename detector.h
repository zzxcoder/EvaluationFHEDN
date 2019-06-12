#ifndef DETECTOR_H
#define DETECTOR_H

#define CPU_ONLY

#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>

class Detector
{
public:
    Detector(const std::string& model_file, const std::string& weight_file,
             const std::string& mean_file, const std::string& mean_value);
    std::vector< std::vector<float> > detect(const cv::Mat& image);         // 检测单张图像

private:
    void setMean_(const std::string& mean_file, const std::string& mean_value);     // 设置均值
    void preprocess_(const cv::Mat& image, std::vector<cv::Mat>* input_channels);
    void wrapInputLayer_(std::vector<cv::Mat>* input_channels);

   boost::shared_ptr< caffe::Net<float> > net_;         // FHEDN网络
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

#endif // DETECTOR_H
