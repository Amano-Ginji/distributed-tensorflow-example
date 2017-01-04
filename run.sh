#!/bin/bash

. ./shflags

data="data"

mkdir -p $data

DEFINE_string 'job_name' 'ps' 'job name, ps or worker' 'j'
DEFINE_integer 'task_index' '0' 'task index' 'i'
DEFINE_string 'train' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train' 'train file path' 't'
DEFINE_string 'test' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test' 'test file path' 'T'
DEFINE_string 'train_file_list' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train_file_list' 'train file list'
DEFINE_string 'test_file_list' 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test_file_list' 'test file list'

function get_file_list() {
  input_path=$1
  file_list=$2
  file_type=$3
  
  hadoop fs -test -e ${file_list} && \
    hadoop fs -rm -r ${file_list}

  hadoop fs -ls ${input_path} | \
    awk -F" " 'NF>7{print $8;}' > \
    $data/${file_type}_file_list

  hadoop fs -put $data/${file_type}_file_list ${file_list}
}

get_file_list ${FLAGS_train} ${FLAGS_train_file_list} "train"
get_file_list ${FLAGS_test} ${FLAGS_test_file_list} "test"



