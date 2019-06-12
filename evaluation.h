#ifndef EVALUATION_H
#define EVALUATION_H

#include "detector.h"

void evaluateFDDBList(Detector* detector);                  // 评估FDDB数据集
void evaluateAFWList(Detector* detector);                   // 评估AFW数据集
void evaluatePascalList(Detector* detector);                // 评估PASCAL数据集
void evaluateWIDERFACEList(Detector* detector);     // 评估WIDERFACE数据集
void detectImage(Detector* detector);       // 检测人脸并可视化结果

#endif // EVALUATION_H
