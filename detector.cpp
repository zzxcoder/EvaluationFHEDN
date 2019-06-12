#include "detector.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

Detector::Detector(const string& model_file, const string& weight_file,
                   const string& mean_file, const string& mean_value)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weight_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 1 || num_channels_ == 3) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    setMean_(mean_file, mean_value);
}

vector< vector<float> > Detector::detect(const Mat &image)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    vector<Mat> input_channels;
    wrapInputLayer_(&input_channels);

    preprocess_(image, &input_channels);

    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* output_data = output_layer->cpu_data();
    const int num_det = output_layer->height();
    vector< vector<float> > detections;
    for (int k = 0; k < num_det; k++) {
        if (output_data[0] == -1) {
            output_data += 7;       // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            continue;
        }

        vector<float> detection(output_data, output_data + 7);
        detections.push_back(detection);
        output_data += 7;
    }

    return detections;
}

void Detector::setMean_(const string &mean_file, const string &mean_value)
{
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

        vector<Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; i++) {
            Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        Mat mean;
        cv::merge(channels, mean);

        channel_mean = cv::mean(mean);
        mean_ = Mat(input_geometry_, mean.type(), channel_mean);
    }

    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) << "Specify either 1 mean_value or as many as channels: " << num_channels_;

        vector<Mat> channels;
        for (int i = 0; i < num_channels_; i++) {
            Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

void Detector::wrapInputLayer_(std::vector<Mat> *input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); i++) {
        Mat channel(height, width, CV_32FC1, data);
        input_channels->push_back(channel);
        data += width * height;
    }
}

void Detector::preprocess_(const Mat &image, std::vector<Mat> *input_channels)
{
    Mat sample;
    if (image.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(image, sample, COLOR_GRAY2BGR);
    else if (image.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(image, sample, COLOR_BGR2GRAY);
    else if (image.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(image, sample, COLOR_BGRA2GRAY);
    else if (image.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(image, sample, COLOR_BGRA2BGR);
    else
        sample = image;

    Mat image_resize;
    if (image_resize.size() != input_geometry_)
        cv::resize(sample, image_resize, input_geometry_);
    else
        image_resize = sample;

    Mat image_float;
    if (num_channels_ == 3)
        image_resize.convertTo(image_float, CV_32FC3);
    else
        image_resize.convertTo(image_float, CV_32FC1);

    Mat sample_normalize;
    cv::subtract(image_float, mean_, sample_normalize);

    cv::split(sample_normalize, *input_channels);
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}
