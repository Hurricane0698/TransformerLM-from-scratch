from cs336_basics.Training import save_checkpoint, load_checkpoint, data_loading
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.loss_optimizer import cross_entropy, gradient_clipping, learning_rate_schedule, AdamW
import argparse
import numpy as np
import torch
from pathlib import Path
import json
import csv
import time
'''
语义层
输入：path, data, 超参数集：模型超参数、优化器超参数、训练数据集设置
功能：模型训练、周期性记录状态、状态保存，从恢复节点继续
输出：把模型数据保存到path

状态变化：
data在变（每次传入数据不一样）
模型状态在变：可学习参数梯度更新
优化器状态在变：一阶矩、二阶矩
训练步数在变：到哪一步了
学习率在变：余弦退火
loss在变
模型看到的token数量在变


不变量清单：
model、data、优化器的数据必须在同一个device上
data必须保证输入和预测目标确定对应、每次只载入这一步的训练内容
参数梯度必须没有被污染，每一步都是由这一步forward得到
优化器必须是当前梯度、当前训练步数计算的最新内容
训练步数必须忠实反映当前状态
学习率必须和训练步数对应
loss评估当前状态，永不参与模型训练，验证集必须分开验证
模型看到的token数量必须由输入的历史累计token总数符合

数据流:np.load mmap的训练数据->每一步采样数据训练；验证阶段为train和validation分别采样计算loss损失
计算流：
对于训练一次step，先进入训练模式（让所有参数可更新），清空上一步的梯度，对这一轮输入前向传播，计算loss，反向传播，然后optimizer更新
对于评估，进入评估模式，然后不带梯度的前向传播，最后计算损失
状态：同状态变化
观察流：
每隔一段时间，
记录训练进度：步数在哪，学习率是否正常（用于诊断异常情况），在train集和validation集的loss情况，savepoint保存点状态显性提示（用于恢复特定历史状态）
，记录看到的token数量(这个没有保存，看不了)，用于绘图

恢复：用loadcheckpoint函数载入特定路径的数据，拿到训练步数，继续训练
配置：用argparse构建命令行参数'''

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model."
    )

    # -------------------------
    # experiment settings
    # -------------------------

    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--random-seed", type=int, required=True)

    # -------------------------
    # Data paths
    # -------------------------
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to the training dataset, e.g. a .npy file loaded with np.memmap.",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        required=True,
        help="Path to the validation dataset, e.g. a .npy file loaded with np.memmap.",
    )

    # -------------------------
    # Model hyperparameters
    # -------------------------
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # -------------------------
    # Optimizer hyperparameters
    # -------------------------
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # -------------------------
    # Training settings
    # -------------------------
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")

    # Number of batches used to estimate train/valid loss during eval
    parser.add_argument("--eval-iters", type=int, default=10)

    # Max l2 norm used by gradient clipping
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # -------------------------
    # Checkpointing
    # -------------------------
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="checkpoint.pt",
        help="Path to save checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=str,
        default=None,
        help="Optional path to load a checkpoint and resume training.",
    )

    return parser

def main(args):
    #log资源创建
    with open(run_dir/"metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "elapsed_seconds", "split", "loss", "learning_rate"]#确定csv的列名
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        #先实例化模型和优化器，优化器学习率不变量，事务在更新参数前完成，必须反映当前步数的学习率
        torch.manual_seed(args.random_seed)
        model = TransformerLM(args.vocab_size, args.context_length, args.d_model,
                                args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
        device = args.device
        model = model.to(device)
        optimizer = AdamW(model.parameters(), args.lr, (args.beta1, args.beta2), args.weight_decay, args.eps)
        #准备数据集，先从路径memmp载入训练和验证数据
        train_data = np.load(args.train_data, 'r')
        valid_data = np.load(args.valid_data, 'r')
        #training 参数
        target_steps = args.num_iters
        batch_size = args.batch_size
        context_length = args.context_length
        eval_interval = args.eval_interval
        save_interval = args.save_interval
        eval_iters = args.eval_iters
        warmup_iters = args.warmup_iters
        max_lr = args.lr
        min_lr = args.min_lr
        save_path = args.checkpoint_out
        max_l2_norm = args.max_grad_norm
        assert warmup_iters < target_steps, "warmup_iters must smaller than num_iters"
        assert eval_iters > 0, "eval_iters must be positive"
        assert eval_interval > 0, "eval_interval must be positive"
        if args.checkpoint_in is not None:
            global_steps = load_checkpoint(args.checkpoint_in, model, optimizer)
        else:
            global_steps = 0
        #开始训练循环
        start_time = time.perf_counter()
        while global_steps < target_steps:
            #事务更新optimizer学习率
            lr_t = learning_rate_schedule(global_steps, max_lr, min_lr, warmup_iters, target_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr_t
            #训练
            model.train()
            model.zero_grad()
            inputs, targets = data_loading(train_data, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm)
            optimizer.step()
            global_steps += 1
            #评估
            if global_steps % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_total_loss = 0
                    valid_total_loss = 0
                    for _ in range(eval_iters):
                        train_inputs, train_targets = data_loading(train_data, batch_size, context_length, device)
                        valid_inputs, valid_targets = data_loading(valid_data, batch_size, context_length, device)
                        train_total_loss += cross_entropy(model(train_inputs), train_targets)
                        valid_total_loss += cross_entropy(model(valid_inputs), valid_targets)
                    train_loss = train_total_loss / eval_iters
                    valid_loss = valid_total_loss / eval_iters
                print(f"Step:{global_steps}", f"Train Loss:{train_loss}", f"Valid Loss:{valid_loss}", f"Learning Rate:{lr_t}")
                elapsed_seconds = time.perf_counter() - start_time
                writer.writerow({"step": global_steps,
                                    "elapsed_seconds": elapsed_seconds,
                                    "split": "train",
                                    "loss": train_loss.item(), #scalar tensor转python float
                                    "learning_rate": lr_t})
                writer.writerow({"step": global_steps,
                                    "elapsed_seconds": elapsed_seconds,
                                    "split": "valid",
                                    "loss": valid_loss.item(),
                                    "learning_rate": lr_t})
                f.flush()#防止中途挂了前面信息全部丢失
            #保存
            if global_steps % save_interval == 0:
                save_checkpoint(model, optimizer, global_steps, save_path)
                print(f"Save state to {save_path}")
        save_checkpoint(model, optimizer, target_steps, save_path)
        print(f"Save state to {save_path}, training finish")

if __name__ == "__main__":
    args = build_argparser().parse_args()
    config = vars(args)
    run_dir = Path("cs336/experiments/")/args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir/"config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    main(args)