cd /home/work/SSD/caffe
./build/tools/caffe train \
--solver="models/FHEDN_512x512/solver.prototxt" \
--weights="models/finetune_VGG/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/FHEDN_512x512/VGG_WIDERFACE_FHEDN_512x512.log
