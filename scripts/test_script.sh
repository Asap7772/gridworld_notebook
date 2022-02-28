MAZE_TYPES=(2 3 4 5 6);
ENV_TYPE=smooth;
DES_TYPE=13;
CQL_ALPHA_VALS=(.1)
TEMPERATURE_VALS=(.1)
dataset_composition=(random+optimal)

for i in "${MAZE_TYPES[@]}";
do
    for j in "${CQL_ALPHA_VALS[@]}";
    do
        for k in "${TEMPERATURE_VALS[@]}";
        do
            for l in "${dataset_composition[@]}";
            do
                echo "Running maze type $i, cql_alpha $j, temperature $k, dataset composition $l"
                python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld_fixedwandb.py --maze_type=$i --env_type=$ENV_TYPE --dataset_composition=$l --cql_alpha_val=$j --transform_type=$DES_TYPE --const_transform=$k  --exp_start test
            done
        done
    done
done



