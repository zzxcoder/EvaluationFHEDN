# EvaluationFHEDN
The face detector uses deep convolutional neural network built on Caffe.  
1、Please install SSD-Caffe (https://github.com/weiliu89/caffe/tree/ssd), and then prepare the training dataset, i.e. WIDER FACE (http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html). The testing datasets include AFW, FDDB, PASCAL and WIDER FACE val.

2、The training script files are saved in the folder named as models/FHEDN_512x512. The weight of network trained by us is available on https://pan.baidu.com/s/1V4C4R2pobGdFSHjZXzhAHw (Extraction code: owj7)

3、The project is estabilshed by Qt Creator 5.7, you also can build it using cmake and please configure it by yourself. Additionally, you should modify the path of testing set in the code for fitting your environment.

If you think it is useful for your work, please cite our work as following:
(1) IJCNN 2018 conference: Z. Zhou, Z. He, C. Ziyu, Y. Jia, H. Wang, J. Du, D. Chen, L. Wang, J. Chen, Fhedn: A context modeling feature hierarchy encoder-decoder network for face detection, in: IEEE International Joint Conference on Neural Networks, 2018, pp. 1–8.

OR

(2) Extent journal paper: Z. Zhou, Z. He, Y. Jia, J. Du, L. Wang, Z. Chen. Context Prior-based with Residual Learning for Face Detection: A Deep Convolutional Encoder-Decoder Network, submit to Signal Processing: Image Communication (under revision). 

Thanks for your attention!
