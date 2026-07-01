
source ~/miniconda3/bin/activate ANN
trial="two_hidd_layer_construc"

mkdir -p $trial

for act in selu hardswish mish; do
  for loss in sce; do
    echo "Running: activation=$act loss=$loss"
    python run_mlp.py \
    --activation $act \
    --loss $loss \
    --hidden1 256 \
    --hidden2 128 \
    --lr 0.01 \
    --wd 0.001 \
    --mom 0.1 \
    --batch 100 \
    --epoch 200 \
    --trial $trial
  done
done