export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=DSPNet

root_path_name=./dataset/ETT-small
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021
#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --seq_len $seq_len \
#  --pred_len 96 \
#  --model_id $model_id_name_$seq_len'_96' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 7 \
#  --c_out 7 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 15 \
#  --patience 5 \
#  --itr 1 --batch_size 128 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --seq_len $seq_len \
#  --pred_len 192 \
#  --model_id $model_id_name_$seq_len'_192' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 7 \
#  --c_out 7 \
#  --n_heads 4 \
#  --d_model 16 \
#  --d_ff 32 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 5 \
#  --itr 1 --batch_size 128 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --model_id $model_id_name_$seq_len'_336' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 7 \
#  --c_out 7 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 5 \
#  --lradj 'TST' \
#  --pct_start 0.4 \
#  --itr 1 --batch_size 128 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --seq_len $seq_len \
#  --pred_len 720 \
#  --model_id $model_id_name_$seq_len'_720' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 7 \
#  --c_out 7 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 5 \
#  --itr 1 --batch_size 128 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log