#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
VOC_DIR=$script_dir/../../

# Directory that contains all wav files
db_root=./data

stage=0123456
#stage 0 : data preperation
#stage 1 : data preprocessing
#stage 2 : wavenet training

#######################################
#          FEATURE SETTING            #
#######################################
input_type="mulaw-quantize" 
quantize_channels=256
sampling_frequency=16000    # sampling frequency
highpass_cutoff=70          # highpass filter cutoff frequency (if 0, will not apply)
n_fft=1024                  # fft length
hop_length=512              # hop length
n_mels=80                   # number of mel bands to generate
num_workers=0               # number of parallel jobs (if 0, will automatically count)

#######################################
#          TRAINING SETTING           #
#######################################
use_gpu=false
number_gpus=0
n_quantize=256
n_aux=80
n_resch=512
n_skipch=256
dilation_depth=10
dilation_repeat=3
kernel_size=2
upsampling_factor=512
use_speaker_code=false
lr=1e-4
weight_decay=0.0
batch_length=20000
batch_size=1
iters=30000
checkpoint_interval=500
log_interval=200
seed=1
resume=""

#######################################
#          DECODING SETTING           #
#######################################
checkpoint=""
config=""
outdir=""
decode_batch_size=1
mode="sampling"


# experiment tag
tag=""

train_set="train"
test_set="test"
datasets=($train_set $test_set)

# experiment name
if [ -z ${tag} ]; then
    expname=${spk}_${train_set}
else
    expname=${spk}_${train_set}_${tag}
fi


# This enable argparse-like parsing of the above variables e.g. ./run.sh --stage 0
. parse_options.sh || exit 1;


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail


# stage 0
# mk_dataset.py
if echo $stage | grep -q 0; then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    # data check
    if [ ! -e $db_root ];then
        echo "ERROR: DB ROOT $db_root must be exist"
        exit 1
    fi

    if [ -z $db_root ]; then
        echo "ERROR: DB ROOT must be specified for train/test splitting."
        echo "Use --db_rood \${path_contains_wav_files}"
        exit 1
    fi

    # directory check
    [ ! -e ${db_root}/local ] && mkdir -p ${db_root}/local
    [ ! -e ${db_root}/${train_set} ] && mkdir -p ${db_root}/${train_set}
    [ ! -e ${db_root}/${test_set} ] && mkdir -p ${db_root}/${test_set}

    python mk_dataset.py \
        --data_dir $db_root/wavs --local_dir ${db_root}/local \
        --train_dir ${db_root}/${train_set} --test_dir ${db_root}/${test_set}
fi

# stage 1
# preprocessing.py
if echo $stage | grep -q 1; then
    echo "###########################################################"
    echo "#                Data PreProcessing Step                  #"
    echo "###########################################################"

    # directory check
    [ ! -e ${db_root}/wavs_pre ] && mkdir -p ${db_root}/wavs_pre
    
    for set in ${datasets[@]}; do 
        python preprocessing.py \
            --data_dir ${db_root}/${set}/wav.scp --data_type ${set} \
            --out_dir ${db_root}/${set} \
            --out_files_dir ${db_root}/wavs_pre \
            --num_workers $num_workers \
            --input_type $input_type \
            --quantize_channels $quantize_channels \
            --sampling_frequency $sampling_frequency \
            --highpass_cutoff $highpass_cutoff \
            --n_fft $n_fft \
            --hop_length $hop_length \
            --n_mels $n_mels
    done
fi 

# stage 2
# set variables
timestamp=`date '+%Y%m%d_%H:%M'`
if [ ! -n "${tag}" ];then
    expdir=exp/${timestamp}_tr
else
    expdir=exp/tr_${timestamp}_tr_${tag}
fi

if echo ${stage} | grep -q 2; then
    echo "###########################################################"
    echo "#               WAVENET TRAINING STEP                     #"
    echo "###########################################################"

    python train.py \
        --datadir ${db_root}/${train_set}/wav-pre.scp \
        --expdir $expdir \
        --use_gpu $use_gpu \
        --number_gpus $number_gpus \
        --n_quantize $n_quantize \
        --n_aux $n_aux \
        --n_resch $n_resch \
        --n_skipch $n_skipch \
        --dilation_depth $dilation_depth \
        --dilation_repeat $dilation_repeat \
        --kernel_size $kernel_size \
        --upsampling_factor $upsampling_factor \
        --use_speaker_code $use_speaker_code \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_length $batch_length \
        --batch_size $batch_size \
        --iters $iters \
        --sr $sampling_frequency \
        --checkpoint_interval $checkpoint_interval \
        --log_interval $log_interval \
        --seed $seed \
        --resume $resume \
        --mode $mode
fi

# stage 3
# set variables
if [ ! -n "${outdir}" ];then
    outdir=exp/${timestamp}_tr/test_output
fi

if echo ${stage} | grep -q 3; then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"

    python decode.py \
        --featdir ${db_root}/${test_set}/wav-pre.scp \
        --checkpoint $checkpoint \
        --config $config \
        --outdir $outdir \
        --sr $sampling_frequency \
        --batch_size $decode_batch_size \
        --mode $mode \
        --use_gpu $use_gpu \
    
fi

    



