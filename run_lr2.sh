#!/bin/bash

source ./shflags

data="data"

mkdir -p $data

DEFINE_string 'job_name' 'ps' 'job name, ps or worker' 'j'
DEFINE_integer 'task_index' '0' 'task index' 'i'
DEFINE_string 'train' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train' 'train file path' 't'
DEFINE_string 'test' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test' 'test file path' 'T'
DEFINE_string 'train_file_list' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train_file_list' 'train file list'
DEFINE_string 'test_file_list' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test_file_list' 'test file list'
DEFINE_string 'mode' 'product' 'run mode, product or test'
DEFINE_string 'output' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/output' 'output root hdfs path'

# parse the command-line
FLAGS "$@" || exit 1
eval set -- "${FLAGS_ARGV}"
if [[ ${print_help} == true ]];then
    ./${0} --help
    exit 1
fi
set -x

:<<comment
function repartition() {
  input_path=$1
  output_path=${FLAGS_output}/train
}
comment

function get_file_list() {
  input_path=$1

  hadoop fs -ls -R ${input_path} | \
    awk -F" " 'BEGIN{ORS=","} NF>7&&$5>1000{print $8;}'

}

train_data=$(get_file_list ${FLAGS_train})
test_data=$(get_file_list ${FLAGS_test})

echo "train data: ${train_data}"
echo "test data: ${test_data}"

if [ ${FLAGS_mode} == 'product' ]; then
  /usr/local/python-2.7.2/bin/tf_tool -a "LR" -c lr2.json lr2.py \
    --batch_size=500 --num_epochs=50 --features=1000000000 \
    --thread_num=8 --learning_rate=1 --trace_step_interval=10000 \
    --train=${train_data} \
    --test=${test_data} \
    --train_sampling_rate=0.01 \
    --test_sampling_rate=0.002
else
  HADOOP_HDFS_HOME="/usr/local/hadoop" python lr2.py \
    --job_name=${FLAGS_job_name} --task_index=${FLAGS_task_index} \
    --train="${train_data}" --test="${test_data}"
fi


########
# Call
########
#sh run_lr2.sh --train="hdfs://10.50.64.84:8020/user/admin/train_data/data" --test="hdfs://10.50.64.84:8020/user/admin/test_data/part-00000" --train_file_list="hdfs://10.50.64.84:8020/user/datamining/yaowq/test/data/train_file_list" --test_file_list="hdfs://10.50.64.84:8020/user/datamining/yaowq/test/data/test_file_list"

#hdfs://10.40.170.151:8020/user/iflow/zhangq/online-ctr_model-batch_half/train_conv/

#sh run_lr2.sh --train="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl_dn_2016122106" --test="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl_dn_2016122107" --train_file_list="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl/data_list/train_file_list_2016122106" --test_file_list="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl/data_list/test_file_list_2016122106"

#hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/train_conv/2017-01-06*
#hadoop distcp hdfs://10.40.170.151:8020/user/iflow/zhangq/online-ctr_model-batch_half/train_conv hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/

#sh run_lr2.sh --train="hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/train_conv/2017-01-06*/" --test="hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/train_conv/2017-01-07*"
#sh run_lr2.sh --train="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl_dn_2016122106" --test="hdfs://10.50.64.84:8020/user/datamining/yaowq/ftrl_dn_2016122107"

#sh run_lr2.sh --train="hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/train_conv2/2017-01-09/" --test="hdfs://10.50.64.84:8020/user/datamining/yaowq/data/ctr/train_conv2/2017-01-10/"
