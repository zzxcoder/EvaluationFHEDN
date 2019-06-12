QT += core
QT -= gui

CONFIG += c++11

TARGET = EvalutionFHEDN
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    detector.cpp \
    evaluation.cpp

INCLUDEPATH += /usr/local/include \
                              /usr/local/include/opencv \
                             /usr/local/include/opencv2
LIBS += -L/usr/local/lib -lopencv_core \
              -L/usr/local/lib -lopencv_highgui \
              -L/usr/local/lib -lopencv_imgproc

INCLUDEPATH += /usr/include \
                              /usr/include/boost \
                              /usr/include/glog \
                             /usr/include/glags

LIBS += -L/usr/lib/x86_64-linux-gnu -lboost_system -lboost_thread -lglog -lgflags -llmdb -lleveldb -lcblas -latlas

INCLUDEPATH += /home/zzx/work/SSD/caffe/include
LIBS += -L/home/zzx/work/SSD/caffe/build/lib -lcaffe

HEADERS += \
    detector.h \
    evaluation.h
