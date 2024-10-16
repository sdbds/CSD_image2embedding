

$train_data_dir = "./2b" # input images path | 图片输入路径
$output_name = "2b"

accelerate launch --num_cpu_threads_per_process=8 "./lancedatasets.py" `
  $train_data_dir `
  --output_name=$output_name