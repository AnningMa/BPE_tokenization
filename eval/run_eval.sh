#!/bin/bash

DEF="y"
read -e -p "Run full experiment [Y/n/q]:" FULL
FULL="${FULL:-${DEF}}"

if [[ $FULL == "q" ]]; then
  exit
elif [[ $FULL == "y" ]]; then
  VocabSizes=('10000' '20000')
  Thresholds=('100' '500')
  Tokenizers=('bpe' 'wpc')
  TrainDatas=('wiki' 'guten600' 'guten1k2')
  for t in "${Tokenizers[@]}"; do
    for d in "${TrainDatas[@]}"; do
      for v in "${VocabSizes[@]}"; do
        if [[ $t == 'wpc' ]]; then
          for m in "${Thresholds[@]}"; do
            python3 ./eval/eval.py --type $t --data $d -v $v -m $m
          done
        else
          python3 ./eval/eval.py --type $t --data $d -v $v
        fi
      done
    done
  done
else
  read -e -p "Which tokenizer [bpe/wpc]:" t
  read -e -p "Which training data [wiki/guten600/guten1k2]:" d
  read -e -p "Vocab size [10000/20000]:" v
  if [[ $t == 'wpc' ]]; then
    read -e -p "Min pair freq threshold [100/500]:" m
    python3 ./eval/eval.py --type $t --data $d -v $v -m $m
  else
    python3 ./eval/eval.py --type $t --data $d -v $v
  fi
fi
