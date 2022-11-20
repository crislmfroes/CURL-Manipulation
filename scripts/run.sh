CUDA_VISIBLE_DEVICES=0 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python train.py \
    --domain_name habitat \
    --task_name HabitatPick-v1 \
    --encoder_type multi_input \
    --action_repeat 8 \
    --save_tb  --save_model --save_video --pre_transform_image_size 76 --image_size 64 \
    --work_dir ./tmp/habitat-pick \
    --agent curl_sac --frame_stack 4 \
    --encoder_feature_dim 50 --n_envs 1 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 1000 --replay_buffer_capacity 50000 --init_steps 1000 --batch_size 128 --num_train_steps 100000 --num_eval_episodes 10