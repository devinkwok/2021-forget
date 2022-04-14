import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from forget.job import Job
from forget.plot.plotter import Plotter
from forget.metrics.transforms import signed_prob, softmax


def evaluate_trained_model(job: Job):
    # evaluate model on ALL train and test data
    train_data = DataLoader(job.get_train_dataset(), batch_size=2500, num_workers=1)
    test_data = DataLoader(job.get_test_dataset(), batch_size=2500, num_workers=1)
    train_s_prob, test_s_prob = {}, {}
    for replicate_dir, subdir in job.replicates():
        ckpt = torch.load(
            os.path.join(replicate_dir, f"ckpt-ep{job.n_epochs}.pt"),
            map_location=torch.device("cpu"),
        )
        model = job.get_model(ckpt["model_state_dict"])
        train_s_prob[subdir] = job.cached(
            lambda: eval(model, train_data),
            subdir,
            f"all_train-signed_prob-ep{job.n_epochs}.pt",
            to_cpu=True,
        )
        test_s_prob[subdir] = job.cached(
            lambda: eval(model, test_data),
            subdir,
            f"all_test-signed_prob-ep{job.n_epochs}.pt",
            to_cpu=True,
        )
    plot_eval_acc(job, train_s_prob, "all-train-acc")
    plot_eval_acc(job, test_s_prob, "all-test-acc")


def eval(model, dataloader: DataLoader):
    s_prob = []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X.cuda()).cpu()
            s_prob.append(signed_prob(softmax(logits), np.array(y)))
    s_prob = np.concatenate(s_prob, axis=0)
    return s_prob


def plot_eval_acc(job, s_prob, plot_name):
    metrics = {}
    for name, s_prob in s_prob.items():
        qq_curve, acc = order_by_accuracy(s_prob)
        metrics[f"{name}-{acc * 100:0.2f}%"] = qq_curve
    plotter = Plotter(job, f"train_plots-ep{job.n_epochs}")
    plotter.plot_array(plotter.plt_curves, plot_name, row_metrics=metrics, width=20)


def order_by_accuracy(s_prob):
    correct_mask = s_prob > 0
    correct = s_prob[correct_mask]
    incorrect = s_prob[np.logical_not(correct_mask)]
    sorted_correct = np.flip(np.sort(correct))
    sorted_incorrect = np.sort(incorrect)
    sorted = np.concatenate([sorted_correct, sorted_incorrect])
    accuracy = np.sum(correct_mask) / len(correct_mask)
    return sorted, accuracy


def plot_training(job: Job):
    n_iters = job.n_epochs * job.n_iter_per_epoch
    train_metrics = {}
    for replicate_dir, name in job.replicates():
        train_loss = np.empty(n_iters)
        train_acc = np.empty(n_iters)
        lr = np.empty(n_iters)
        for j in range(job.n_epochs):
            epoch = j + 1  # epoch 0 is ckpt at initialization, skip
            ckpt = torch.load(
                os.path.join(replicate_dir, f"ckpt-ep{epoch}.pt"),
                map_location=torch.device("cpu"),
            )
            start = j * job.n_iter_per_epoch
            end = start + job.n_iter_per_epoch
            train_loss[start:end] = ckpt["loss"]
            train_acc[start:end] = ckpt["train accuracy"]
            lr[start:end] = ckpt["optimizer_state_dict"]["param_groups"][0]["lr"]
        log_lr = -1 * np.log10(lr) / 10
        train_metrics[name] = np.stack([train_loss, train_acc, log_lr], axis=0)

    plotter = Plotter(job, f"train_plots-ep{job.n_epochs}")
    group = np.repeat(np.arange(3).reshape(-1, 1), n_iters, axis=1)
    plotter.plot_smooth_curves(
        "Training",
        train_metrics,
        group=group,
        group_names=["Loss", "Accuracy", "-log_10(LR)/10"],
        smoothing=job.n_iter_per_epoch,
        hlines=[1.0],
        vlines=[79 * job.n_iter_per_epoch, 119 * job.n_iter_per_epoch],
    )
