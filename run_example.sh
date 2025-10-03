#!/bin/sh

run_train=true
run_attack=true
iterations="1"

# regularization
regularizer_betas="0 0.075"

attack_f_1="fgsm"
epsilons_1="0.05 0.1 0.3"

attack_f_2="jitter"
epsilons_2="0.035 0.045"

attack_f_3="mixup"
epsilons_3="0.025 0.1 0.75"

models="vgg16"
target_model="vgg16"
dataset=1
crop_size=224

# DLRT parameters
dlrt="0" # 0 = baseline training
tolerances="0.01"
rmax="200"
num_local_iter=10
init_r=150

# training parameters
train_batch_size=16
val_batch_size=128
num_epochs=20
num_epochs_ft=1

# logging
wandb=1
wandb_tag="example"

# data root
data_root="./dataset/data_adversarial_rs/"

# decide folder based on dlrt flag
if [ "$dlrt" = "0" ]; then
    subdir="baseline"
else
    subdir="low_rank"
fi

# pick dataset folder
case "$dataset" in
    6) ds_name="ImageNet_adv" ;;
    3) ds_name="Cifar10_adv" ;;
    2) ds_name="AID_adv" ;;
    1) ds_name="UCM_adv" ;;
    *) echo "Unknown dataset id: $dataset" ; exit 1 ;;
esac

# helper to delete old adversarial images
delete_adversarial_images() {
    attack="$1"
    model="$2"
    target="./dataset/data_adversarial_rs/$ds_name/$attack/$subdir/$model/*.png"
    echo "Deleting $target"
    rm -f $target
}

# helper to run attack + test
run_attack_test() {
    attack="$1"
    epsilons="$2"
    model="$3"
    tol="$4"
    beta="$5"

    for eps in $epsilons; do
        delete_adversarial_images "$attack" "$model"

        python generate_attacked_images.py \
            --dataID $dataset \
            --dlrt $dlrt \
            --rmax $rmax \
            --init_r $init_r \
            --tol "$tol" \
            --robusteness_regularization_beta "$beta" \
            --attack_func $attack \
            --network $model \
            --crop_size $crop_size \
            --attack_epsilon $eps \
            --save_path_prefix $data_root \
            --model_root_dir ./models/ \
            --data_root_dir $data_root

        python test_adversarial_accuracy.py \
            --dataID $dataset \
            --dlrt $dlrt \
            --rmax $rmax \
            --init_r $init_r \
            --tol "$tol" \
            --robusteness_regularization_beta "$beta" \
            --target_network $target_model \
            --surrogate_network $model \
            --attack_func $attack \
            --wandb $wandb \
            --wandb_tag "$wandb_tag" \
            --attack_epsilon $eps \
            --crop_size $crop_size \
            --val_batch_size $val_batch_size \
            --save_path_prefix ./models/ \
            --data_root_dir $data_root
            
    done
}

# main loop
for i in $(seq 1 $iterations); do
    echo "iteration $i"
    for model in $models; do
        for tol in $tolerances; do
            for beta in $regularizer_betas; do

                if [ "$run_train" = true ]; then
                    python compressed_transfer_learning.py \
                        --dataID $dataset \
                        --num_local_iter $num_local_iter \
                        --rmax $rmax \
                        --dlrt $dlrt \
                        --init_r $init_r \
                        --lr 5e-4 \
                        --num_epochs $num_epochs \
                        --num_epochs_low_rank_ft $num_epochs_ft \
                        --network $model \
                        --save_name "$beta" \
                        --tol "$tol" \
                        --wandb $wandb \
                        --wandb_tag "$wandb_tag" \
                        --robusteness_regularization_beta "$beta" \
                        --val_batch_size $val_batch_size \
                        --train_batch_size $train_batch_size \
                        --crop_size $crop_size \
                        --print_per_batches 10 \
                        --load_model 0 \
                        --root_dir $data_root \
                        --num_workers 32 \
                        --save_path_prefix ./models/
                fi

                if [ "$run_attack" = true ]; then
                    run_attack_test "$attack_f_1" "$epsilons_1" "$model" "$tol" "$beta"
                    run_attack_test "$attack_f_2" "$epsilons_2" "$model" "$tol" "$beta"
                    run_attack_test "$attack_f_3" "$epsilons_3" "$model" "$tol" "$beta"
                fi
            done
        done
    done
done

