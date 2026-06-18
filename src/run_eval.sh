#!/bin/bash

VocabSizes=('5000' '10000')
Thresholds=('100' '500' '1000')
Tokenizers=('bpe' 'wpc')
TrainDatas=('wiki' 'guten')

for t in "${Tokenizers[@]}"; do
    for d in "${TrainDatas[@]}"; do
        for v in "${VocabSizes[@]}"; do
            if [[ $t == 'wpc' ]]; then
                for m in "${Thresholds[@]}"; do
                    pixi run python eval.py -w eval -a ${t}_${d}_${v}_${m} --eg
                done
            else
                pixi run  python eval.py -w eval -a ${t}_${d}_${v} --eg
            fi
        done
    done
done

pixi run python eval.py -w eval -a morfessor --eg

pixi run python eval.py -w compare -a
