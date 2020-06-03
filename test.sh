#!/bin/zsh

dir=data/

t=1234

d=123

config=123

train_set="train"
test_set="test"
datasets=($train_set $test_set)

for set in ${datasets[@]}; do
    echo $set
done

timestamp=`date '+%Y/%m/%d_%H:%M'`
echo $timestamp