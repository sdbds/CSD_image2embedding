

$train_data_dir = "./datasets" # input images path | 图片输入路径
$output_name = "datasets"

python "./lancedatasets.py" `
  $train_data_dir `
  --output_name=$output_name