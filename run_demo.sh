#!/usr/bin/env bash

DATASET=cub  #cub flower imagenet_sub cifar100
GPU=0
NET=resnet18 # resnet32
LR=1e-5
EPOCH=101
SAVE=50
LOSS=MSLoss # MSLoss
TASK=10 #6 11
BASE=100 #17
SEED=1

for Method in Finetuning #LwF EWC MAS Finetuning
do
for Tradeoff in 1 # 1 1e7 1e6
do

NAME=${Method}_${Tradeoff}_${DATASET}_${LOSS}_${NET}_${LR}_${EPOCH}epochs_task${TASK}_base${BASE}_seed${SEED}

CUDA_VISIBLE_DEVICES=1 python3 ori_train.py -base ${BASE} -seed ${SEED} -task ${TASK} -data ${DATASET} -tradeoff ${Tradeoff} -exp ${Tradeoff} -net ${NET} -method ${Method} \
-lr ${LR} -dim 256  -num_instances 16 -BatchSize 64 -loss ${LOSS}  -epochs ${EPOCH} -log_dir ${DATASET}_seed${SEED}/${NAME}  \
-save_step ${SAVE} -gpu ${GPU}


for SIGMA_TEST in 0.30 #0.30
do

CUDA_VISIBLE_DEVICES=0 python3 test.py -seed ${SEED} -base ${BASE} -task ${TASK} -epochs ${EPOCH} -data ${DATASET} -gpu ${GPU} -method ${Method} -r  \
checkpoints/${DATASET}_seed${SEED}/${NAME} -mapping_test -sigma_test ${SIGMA_TEST} \
>./results/${DATASET}/${NAME}_SDC_sigma_test${SIGMA_TEST}.txt

done

done
done

