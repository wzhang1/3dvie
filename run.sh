export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 fine-tune.py \
    --gpus 0,1,2,3 \
    --pretrained-model model/resnet-18 \
    --data-train data/image_tr.lst --model-prefix model_level3-35degree/3dvie-resnet-18 \
    --data-val data/image_dev.lst \
	--data-nthreads 48 \
    --batch-size 32 --num-classes 3 --num-examples 16763\
    --lr=.003\
    --lr-factor=0.1\
    --lr-step-epochs='5, 10, 15, 25'\
    --max-random-rotate-angle=15\
    --mom=0.1

#--pretrained-model model/resnet-152 \

