CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
--sft_type "full" \
--model_id_or_path ZhipuAI/chatglm3-6b \
--model_revision master \
--custom_train_dataset_path bopdata_train.json \
--custom_val_dataset_path bopdata_test.json \
--save_steps "20" \
--batch_size "1" \
--learning_rate "1e-04" \
--eval_batch_size "1" \
--output_dir output_chatglm \
--logging_dir output_chatglm \
--num_train_epochs "2" \
--dataset_test_ratio "0.15" \


