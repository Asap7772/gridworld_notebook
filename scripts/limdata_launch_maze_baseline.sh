MAZE_TYPES=(2 3 4 5 6);
ENV_TYPE=smooth;
DES_TYPE=0;
CQL_ALPHA_VALS=(.1 1 2 5 10 20 50 100)
DATASET_COMPOSITION=(random+optimal mixed_limited mixed_limited_skewed)
PREFIX=exp
PROJ_NAME=algo_gridworlds_baseline_data1x
COUNTER=0
NUM_GPUS=$(nvidia-smi -L | wc -l)

for i in "${MAZE_TYPES[@]}"
do
    for j in "${CQL_ALPHA_VALS[@]}"
    do
        for l in "${DATASET_COMPOSITION[@]}"
        do
            echo "Running variant $DES_TYPE with maze type $i, cql_alpha $j, dataset composition $l with on GPU $COUNTER"
            export CUDA_VISIBLE_DEVICES=$COUNTER
            python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld_limited.py --maze_type=$i --env_type=$ENV_TYPE --dataset_composition=$l --cql_alpha_val=$j --transform_type=$DES_TYPE --const_transform=1 --exp_start=$PREFIX --proj_name=$PROJ_NAME  --dataset_size -1 &
            let COUNTER++
            
            if [ $COUNTER -eq $NUM_GPUS ]; then
                wait
                COUNTER=0
            fi
        done
    done
done



