export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py \
    --gpus 0,1,2,3 \
    --pretrained-model model/resnet-52 \
    --data-train data/image_tr23.lst --model-prefix model_23/3dvie-resnet-152 \
    --data-val data/image_dev23.lst \
	--data-nthreads 48 \
    --batch-size 32 --num-classes 2 --num-examples 11255\
    --lr=.001\
    --lr-factor=0.1\
    --lr-step-epochs='5, 10, 15, 25'\
    --max-random-rotate-angle=35


#--pretrained-model model/resnet-152 \

