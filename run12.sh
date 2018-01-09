export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py \
    --gpus 0,1,2,3 \
    --pretrained-model model/resnet-152 \
    --data-train data/image_tr12.lst --model-prefix model_12/3dvie-resnet-152 \
    --data-val data/image_dev12.lst \
	--data-nthreads 48 \
    --batch-size 32 --num-classes 2 --num-examples 11040\
    --lr=.0003\
    --lr-factor=0.1\
    --lr-step-epochs='5, 10, 15, 25'


#--pretrained-model model/resnet-152 \

