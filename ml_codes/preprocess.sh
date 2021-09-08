#!/bin/bash

mkdir -p inputs

for file in $(cd ../benchmark_data && ls *.csv) ; do
  echo $file
  if [ ! -e $file ] ; then
    cp ../benchmark_data/$file inputs
  fi
done

./preprocess

head -n2000000 inputs/runs12-peak.csv > inputs/runs12-peak2M.csv

