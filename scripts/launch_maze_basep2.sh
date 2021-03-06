MAZE_TYPE=2;
ENV_TYPE=smooth;
DES_TYPE=0;

#CQL BASELINE

python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=.1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=5 --transform_type=0 --const_transform=1 
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=10 --transform_type=0 --const_transform=1 
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=20 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=50 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=random+optimal --cql_alpha_val=100 --transform_type=0 --const_transform=1 &

python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=.1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=5 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=10 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=20 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=50 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited --cql_alpha_val=100 --transform_type=0 --const_transform=1 &

python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=.1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=1 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=5 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=10 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=20 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=50 --transform_type=0 --const_transform=1 &
python /home/asap7772/asap7772/algo_gridworld/offline_rl_gridworld.py --maze_type=$MAZE_TYPE --env_type=$ENV_TYPE --dataset_composition=mixed_limited_skewed --cql_alpha_val=100 --transform_type=0 --const_transform=1 &
