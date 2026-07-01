# HW1 
## 运行程序

模型参数与架构可以在run_mlp.sh中修改
```bash
# run_mlp.sh 
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
```
在命令行输入：
```bash
bash codes/run_mlp.sh
```

## Modifications
### run_mlp.py
为方便调整超参与模型架构，主程序在原本基础上有所修改  
主要修改如下
```python
def main(args):

    train_data, test_data, train_label, test_label = load_mnist_2d('codes/data')

    # Your model defintion here
    # You should explore different model architecture
    model = Network()
    model.add(Linear("fc1", 784, args.hidden1, 0.01))
    model.add(get_activation(args.activation))
    if args.hidden2 is not None:
        model.add(Linear("fc2", args.hidden1, args.hidden2, 0.01))
        model.add(get_activation(args.activation))
        model.add(Linear("fc3", args.hidden2, 10, 0.01))
    else:
        model.add(Linear("fc2", args.hidden1, 10, 0.01))
    
    loss = get_loss(args.loss)
    arch = f"{args.activation}_{args.loss}_h{args.hidden1}"
    config = {
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "momentum": args.mom,
        "batch_size": args.batch,
        "max_epoch": args.epoch,
        "disp_freq": 50,
        "test_epoch": 5,
    }

```



### utils.py
加入存储loss与acc的函数与绘制曲线函数，在main.py中调用，这里不再做展示。

### solve_net.py
增加了统计每轮训练所有batch的loss与acc平均值的功能，并最终返回数据。