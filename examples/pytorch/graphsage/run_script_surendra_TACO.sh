#!/bin/bash

# Run the command and save the output to a variable
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
batch_sizes=(1024 2048 4096 8192 16384 32768 65536)
# batch_sizes=(1024)
# batch_sizes=(8192)
fanouts=(20)
#fanouts=(20 15 10)
for fanout in "${fanouts[@]}"; do
  # Loop through each batch size
  for batch_size in "${batch_sizes[@]}"; do
    #sampling_time=0.0
    #training_time=0.0
    #spmm_time=0.0
    last_spmm_time=0.0
    last_cuda_sampling_time=0.0
    add_spmm_time=true
    # if [ "$dataset" = "yelp" ]; then
    #   output=$(python3 node_classification_yelp_surendra.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2 --method=$3)
    # else
    #   output=$(python3 node_classification1.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2 --method=$3)
    # fi
    # output=$(python3 node_classification_dissimilarity_cosine_eigenvector_centrality_yelp.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2)
    output=$(python3 node_classification_dissimilarity_cosine_eigenvector_centrality_dominating_set_maintained_train_nodes_yelp.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2)
    #output=$(python3 node_classification1.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2)
    #output=$(python3 node_classification1.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2 --method=$3)
    #output=$(python3 node_classification_yelp_surendra.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fanout=$fanout,$fanout,$fanout --epoch=$2 --method=$3)
    #filename="SPMM_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    filename="time_centrality_EuroPar/dominating_set/training_70/important_trainidx/$1_F${fanout}_B${batch_size}_puregpu_E$2.txt"
    #filename="epoch1_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    #echo "Dataset = $1" > $filename
    echo "Dataset = $1, batch_size = $batch_size" > $filename
    #python3 node_classification.py --dataset=ogbn-products --batch_size=1024
    #Loop through the output lines to calculate sampling time
    # Reverse the output to find the last occurrence quickly
    last_spmm_time=$(echo "$output" | tac | grep -m1 "^spmm time" | awk '{print $3}')
    last_cuda_sampling_time=$(echo "$output" | tac | grep -m1 "^cuda sampling time" | awk '{print $4}')
    echo ""
    echo "last_spmm_time: $last_spmm_time , last_cuda_sampling_time: $last_cuda_sampling_time"
    tail -3 "epoch_data.txt"
    # while read -r line; do
    #   if [[ $line == Testing...* ]]; then
    #     add_spmm_time=false
    #   fi
    #   # Check if the line contains the string "cuda,sapmling"
    #   if [[ $line == spmm\ time* ]] && $add_spmm_time; then
    #     # Extract the time value and add it to the sampling time
    #     #echo $line
    #     #echo "spmm"
    #     # spmm_time_value=$(echo $line | awk '{print $3}')
    #     last_spmm_time=$(echo $line | awk '{print $3}')
    #
    #     #echo $time_value
    #     # spmm_time=$(echo "$spmm_time + $spmm_time_value" | bc -l)
    #     # fi
    #   elif [[ $line == cuda\ sampling\ time* ]]; then
    #     # Extract the time value and add it to the sampling time
    #     #echo $line
    #     #echo "cuda"
    #     # time_value=$(echo $line | awk '{print $4}')
    #     last_cuda_sampling_time=$(echo $line | awk '{print $4}')
    #     #echo $time_value
    #     # sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
    #   fi
    #
    # done <<< "$output"

      # Check if epoch_data.txt exists
      if [ -f "epoch_data.txt" ]; then
        cat "epoch_data.txt" >> $filename
        echo "Data copied successfully!"
      else
        echo "Error: epoch_data.txt does not exist."
      fi
      # echo "Total sampling time :" $sampling_time ", Total training time :" $training_time >> $filename
      # echo "Total spmm time , Total sampling time" >> $filename
      # echo $spmm_time"," $sampling_time >> $filename
      echo "spmm_time, sampling_time" >> $filename
      echo "$last_spmm_time, $last_cuda_sampling_time" >> $filename
      # echo "Total sampling time : " $sampling_time/$epoch ", Total training time :" $training_time/$epoch >> $filename
      # echo $test >> $filename
      # echo $tt_time >> $filename
    done
  done
