#!/bin/bash

source ~/.profile

HADOOP_HDFS_HOME="/usr/local/hadoop" python lr.py --job_name="worker" --task_index=0
