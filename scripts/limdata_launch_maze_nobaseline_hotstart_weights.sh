MAZE_TYPES=(2 3 4 5 6);
ENV_TYPE=smooth;
DES_TYPE=13;
CQL_ALPHA_VALS=(.1 1 2 5 10 20 50 100)
TEMPERATURE_VALS=(.1 1 2 5 10 20 50 100)
DATASET_COMPOSITION=(random+optimal mixed_limited mixed_limited_skewed)
PREFIX=exp
PROJ_NAME=algo_gridworlds_hotstartweights_data1x
COUNTER=0
WHICH_GPUs=(0 1 4 5)
NUM_GPUS=${#WHICH_GPUs[@]}

for i in "${MAZE_TYPES[@]}"
do
    for j in "${CQL_ALPHA_VALS[@]}"
    do
        for k in "${TEMPERATURE_VALS[@]}"
        do
            for l in "${DATASET_COMPOSITION[@]}"
            do
                gpu_num=${WHICH_GPUs[$((COUNTER%NUM_GPUS))]}
                echo "Running variant $DES_TYPE with maze type $i, cql_alpha $j, temperature $k, dataset composition $l with hotstart weights on GPU $gpu_num"
                export CUDA_VISIBLE_DEVICES=$gpu_num
                python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld_limited.py --maze_type=$i --env_type=$ENV_TYPE --dataset_composition=$l --cql_alpha_val=$j --transform_type=$DES_TYPE --const_transform=$k --exp_start=$PREFIX --proj_name=$PROJ_NAME --hotstart_weight  --dataset_size -1 &
                let COUNTER++

                if [ $COUNTER -eq $NUM_GPUS ]; then
                    wait
                    COUNTER=0
                fi
            done
        done
    done
done



