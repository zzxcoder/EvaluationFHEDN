#include "evaluation.h"
#include  <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include <caffe/blob.hpp>
#include <fstream>
#include <string.h>
#include <sys/time.h>

using namespace cv;
using namespace caffe;
using namespace std;

void evaluateFDDBList(Detector* detector) {
   string rootDir = "/media/zzx/Data/FaceData/StdData/FDDB/";
    string listDir(rootDir);
    string imageDir(rootDir);
    imageDir.append("image/originalPics/");
    ifstream infile(listDir.append("FDDB-folds/FDDB-list.txt"));
    string foldFilePath;

    string out_file = "FDDB_FHEDN_512x512.txt";
    streambuf* buf = std::cout.rdbuf();
    ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    ostream out(buf);

    float confidence_threshold = 0.01;
    while (infile >> foldFilePath) {
        string tmp(rootDir);
        ifstream foldFile(tmp.append("FDDB-folds/") + foldFilePath);
        string imgFile;
        cout << tmp + foldFilePath << endl;
        while (foldFile >> imgFile) {
            string imgPath = imageDir + imgFile;
            Mat image = imread(imgPath.append(".jpg"));
            CHECK(!image.empty()) << "Unable to decode image " << imgFile;
            vector< vector<float> > detections = detector->detect(image);

            cout << "Process image: " << imgFile << endl;
            out << imgFile << endl;
            int num = 0;
            vector<string> results;
            for (size_t i = 0; i < detections.size(); i++) {
                const vector<float>& detection = detections[i];
                CHECK_EQ(detection.size(), 7);
                const float score = detection[2];
                if (score > confidence_threshold) {
                    float xmin = detection[3] * image.cols;
                    float ymin = detection[4] * image.rows;
                    float xmax = detection[5] * image.cols;
                    float ymax = detection[6] * image.rows;
                    float width = xmax - xmin + 1.0;
                    float height = ymax - ymin + 1.0;
                    char tmp[200];
                    sprintf(tmp, "%f %f %f %f %f", xmin, ymin, width, height, score);
                    string det(tmp);
                    results.push_back(det);
                    num++;
                }
            }
            cout << num << endl;
            out << num << endl;
            for (size_t j = 0; j < results.size(); j++) {
                cout << results[j] << endl;
                out << results[j] << endl;
            }
        }
    }
}

void evaluateAFWList(Detector* detector) {
   const string fileList = "/media/zzx/Data/FaceData/StdData/AFW/testimages/test.dat";
    string file;

    string out_file = "AFW_FHEDN_512x512.txt";
    streambuf* buf = std::cout.rdbuf();
    ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    ostream out(buf);

    float confidence_threshold = 0.01;
    ifstream infile(fileList);
    while (infile >> file) {
        string imgFile = "/media/zzx/Data/FaceData/StdData/AFW/testimages/";
        cv::Mat image = imread(imgFile.append(file));
        CHECK(!image.empty()) << "Unable to decode image " << file;
        vector< vector<float> > detections = detector->detect(image);
        cout << "Process image: " << file << endl;

        int num = 0;
        vector<string> results;
        string name = file.substr(0, file.find("."));
        for (size_t i = 0; i < detections.size(); i++) {
            const vector<float>& detection = detections[i];
            CHECK_EQ(detection.size(), 7);
            const float score = detection[2];
            if (score > confidence_threshold) {
                float xmin = detection[3] * image.cols;
                float ymin = detection[4] * image.rows;
                float xmax = detection[5] * image.cols;
                float ymax = detection[6] * image.rows;
                char tmp[200];
                sprintf(tmp, "%s %f %f %f %f %f", name.c_str(), score, xmin, ymin, xmax, ymax);
                string det(tmp);
                results.push_back(det);
                num++;
            }
        }
        cout << num << endl;
        for (size_t j = 0; j < results.size(); j++) {
            cout << results[j] << endl;
            out << results[j] << endl;
        }
        if (0 == results.size())
            out << file << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << endl;
    }
}

void evaluatePascalList(Detector* detector) {
    const string fileList = "/media/zzx/Data/FaceData/StdData/PASCAL/image.list";
    string file;

    string out_file = "PASCAL_FHEDN_512x512.txt";
    streambuf* buf = std::cout.rdbuf();
    ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    ostream out(buf);

    float confidence_threshold = 0.01;
    ifstream infile(fileList);
    while (infile >> file) {
        string imgFile = "/media/zzx/Data/FaceData/StdData/PASCAL/JPEGImages/";
        cv::Mat image = imread(imgFile.append(file));
        CHECK(!image.empty()) << "Unable to decode image " << file;
        vector< vector<float> > detections = detector->detect(image);
        cout << "Process image: " << file << endl;

        int num = 0;
        vector<string> results;
        for (size_t i = 0; i < detections.size(); i++) {
            const vector<float>& detection = detections[i];
            CHECK_EQ(detection.size(), 7);
            const float score = detection[2];
            if (score > confidence_threshold) {
                float xmin = detection[3] * image.cols;
                float ymin = detection[4] * image.rows;
                float xmax = detection[5] * image.cols;
                float ymax = detection[6] * image.rows;
                char tmp[200];
                sprintf(tmp, "%s %f %f %f %f %f", file.c_str(), score, xmin, ymin, xmax, ymax);
                string det(tmp);
                results.push_back(det);
                num++;
            }
        }
        cout << num << endl;
        for (size_t j = 0; j < results.size(); j++) {
            cout << results[j] << endl;
            out << results[j] << endl;
        }
        if (0 == results.size())
            out << file << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << endl;
    }
}

void evaluateWIDERFACEList(Detector* detector) {
    const string fileList = "/media/zzx/Data/FaceData/StdData/WIDERFACE/WIDER_val/images/file.list";
    ifstream infile(fileList);
    string file;
    float confidence_threshold = 0.01;
    while (infile >> file) {
        string path = "/media/zzx/Data/FaceData/StdData/WIDERFACE/WIDER_val/images/";
        cv::Mat image = imread(path.append(file));
        CHECK(!image.empty()) << "Unable to decode image " << file;
        vector< vector<float> > detections = detector->detect(image);
        cout << "Process image: " << file << endl;

        int num = 0;
        vector<string> results;
        for (size_t i = 0; i < detections.size(); i++) {
            const vector<float>& detection = detections[i];
            CHECK_EQ(detection.size(), 7);
            const float score = detection[2];
            if (score > confidence_threshold) {
                float xmin = detection[3] * image.cols;
                float ymin = detection[4] * image.rows;
                float xmax = detection[5] * image.cols;
                float ymax = detection[6] * image.rows;
                float width = xmax - xmin + 1;
                float height = ymax - ymin + 1;
                char tmp[200];
                sprintf(tmp, "%f %f %f %f %f", xmin, ymin, width, height, score);
                string det(tmp);
                results.push_back(det);
                num++;
            }
        }

        cout << num << endl;
        string out_file("WIDERFACE_val/pred/");
        file.replace(file.find(".jpg"), 4, ".txt");
        out_file.append(file);
        streambuf* buf = std::cout.rdbuf();
        ofstream outfile;
        if (!out_file.empty()) {
            outfile.open(out_file.c_str());
            if (outfile.good()) {
                buf = outfile.rdbuf();
            }
        }
        ostream out(buf);
        out << file << endl;
        out << num << endl;
        for (size_t j = 0; j < results.size(); j++) {
            cout << results[j] << endl;
            out << results[j] << endl;
        }
    }
}

void detectImage(Detector* detector) {
    const string& file_type = "image";
    const string& out_file = "";

    streambuf* buf = std::cout.rdbuf();
    ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    ostream out(buf);

    float confidence_threshold = 0.5;
    ifstream infile("/home/zzx/work/EvaluationFHEDN/images/pictures.txt");
    string file;
    double time_total = 0;
    int nfile = 0;
    while (infile >> file) {
        if (file_type == "image") {
            Mat image = imread(file);
            CHECK(!image.empty()) << "Unable to decode image " << file;

            struct timeval tv_begin, tv_end;
            gettimeofday(&tv_begin, NULL);
            vector< vector<float> > detections = detector->detect(image);
            gettimeofday(&tv_end, NULL);
            double time_used = 1000000 * (tv_end.tv_sec - tv_begin.tv_sec) + tv_end.tv_usec - tv_begin.tv_usec;
            cout << " Time = "<< time_used/1000 << " ms" << endl;
            time_total += time_used;

            for (size_t i = 0; i < detections.size(); i++) {
                const vector<float>& detection = detections[i];
                CHECK_EQ(detection.size(), 7);
                const float score = detection[2];
                if (score > confidence_threshold) {
                    out << file << " ";
                    out << static_cast<int>(detection[1]) << " ";
                    out << score << " ";
                    int xmin = static_cast<int>(detection[3] * image.cols);
                    int ymin = static_cast<int>(detection[4] * image.rows);
                    int xmax = static_cast<int>(detection[5] * image.cols);
                    int ymax = static_cast<int>(detection[6] * image.rows);
                    out << xmin << " ";
                    out << ymin << " ";
                    out << xmax << " ";
                    out << ymax << endl;
                    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), Scalar(0,0,255), 2);
                }
            }
            imshow("Face", image);
            waitKey(0);
        }
        nfile++;
    }
    cout << "CPU Time avg = " << time_total / 1000 / nfile << " ms." << endl;
}
