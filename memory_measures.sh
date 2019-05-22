#!/bin/bash

#SBATCH --nodes=3
#SBATCH --mem=20000
#SBATCH --time=1:00:00

module load python-3.4.0
module load spark/2.1.0
module load myhadoop-0.30
module load java-1.8.0_40
alias python=python3

#export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.221-2.6.18.0.el7_6.x86_64/jre
export JAVA_HOME=/storage/software/jre-1.8.0_40/
#export HADOOP_CONF_DIR=$HOME/hadoop-conf.$SLURM_JOBID

#spark-submit --master yarn task_binary_classification.py
spark-submit task_binary_classification.py --py-files my_dependency.txt
#spark-submit --version

