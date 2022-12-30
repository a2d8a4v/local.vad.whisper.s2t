#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=10000
data_root=
q_dir=$data_root/q

level=medium

cmd=run.pl
CUDA=0,1,2
nspk=4
max_nj=30
max_nj_cuda=6


. ./utils/parse_options.sh
set -euo pipefail

if [ ! -z $CUDA ]; then
    # See: https://stackoverflow.com/questions/10586153/how-to-split-a-string-into-an-array-in-bash
    readarray -td ',' cuda_array < <(awk '{ gsub(/, /,"\0"); print; }' <<<"$CUDA, ");
    declare -p cuda_array;
    cuda_array_len="${#cuda_array[@]}"
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    echo 'Stage 0. split wav chunk by VAD...'
    mkdir -pv $data_dir/wav_chunk > /dev/null 2>&1
    python vad.py \
        --input_wav_scp_file_path $data_dir/wav.scp \
        --output_dir_path $data_dir/wav_chunk \
        --output_new_wav_scp_path $data_dir/wav_chunk.scp > /dev/null
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    
    echo 'Stage 1. parallel input generating...'
    if [ $nspk -ge $max_nj_cuda ]; then
        nspk=$max_nj_cuda;
    fi

    mkdir -pv $q_dir > /dev/null 2>&1

    # split the list for parallel processing
    wav_chunk=""
    for n in `seq $nspk`; do
        wav_chunk="$wav_chunk $q_dir/wav_chunk.$n.scp"
    done
    utils/split_scp.pl $data_root/wav_chunk.scp $wav_chunk || echo 'Splitting error'
fi

# . .miniconda3/bin/activate
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then
    
    echo 'Stage 2. parallel transcribing...'
    if [ $nspk -ge $max_nj_cuda ]; then
        nspk=$max_nj_cuda;
    fi

    mkdir -pv $q_dir/log > /dev/null 2>&1
    # CUDA_VISIBLE_DEVICES="${cuda_array[$((cuda_array_len % 'JOB'))]}" python run.py \
    $cmd JOB=1:$nspk $q_dir/log/wav_chunk.JOB.log \
        python run.py \
        --input_wav_scp_file_path $q_dir/wav_chunk.JOB.scp \
        --output_text_file_path $q_dir/results_chunk.JOB.$level.txt \
        --level $level
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then

    echo 'Stage 3. combine outputs...'
    cat $(find $q_dir -name "results_chunk.*.$level.txt" | sort -u | tr '\n' ' ') > $data_root/results_chunk.$level.txt
    if [ $(wc -l $data_root/results_chunk.$level.txt) -nq $($data_dir/wav_chunk.scp) ]; then
        echo "Validation failed, you lose some utterances..."
    fi
fi