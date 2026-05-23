#!/bin/bash


batch_pool=("32" "16" "8" "4")
lr_pool=("0.00001" "0.0001" "0.000001")
opt_pool=("Lion" "Adam" "AdamW" "RMSprop")
views=("1" "2" "3")

for view in ${views[@]}
    do
    for batch in ${batch_pool[@]}
        do
        for lr in ${lr_pool[@]}
            do
            for opt in ${opt_pool[@]}
                do
                    python main_cont.py "$batch" "$lr" "$opt" "$view"
                done
            done
        done
    done
