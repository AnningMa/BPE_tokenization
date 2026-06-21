#!/bin/bash


VocabSizes=('10000' '20000')
Thresholds=('100' '500')
Tokenizers=('bpe' 'wpc')
TrainDatas=('wiki' 'guten600' 'guten1k2')

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

#pixi run python eval.py -w eval -a morfessor --eg

#pixi run python eval.py -w compare -a
