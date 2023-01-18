import time
from datetime import timedelta
import yaml
import argparse
import shutil
import os
import logging
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from utils.builder import build_model, build_loader, build_transforms, build_optim
from utils.util import set_seed, count_parameters, AverageMeter, reduce_mean, save_model, simple_accuracy
from utils.dist import get_world_size


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="config/example.yaml",
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r", encoding='utf-8') as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


def run(hyperparams):
    print(hyperparams)

    # Setup CUDA, GPU & distributed training
    if hyperparams["train"]["local_rank"] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(hyperparams["train"]["local_rank"])
        device = torch.device("cuda", hyperparams["train"]["local_rank"])
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        n_gpus = 1

    hyperparams["train"]['device'] = device
    hyperparams["train"]['nprocs'] = torch.cuda.device_count()

    # Create logging
    logging_name = hyperparams['train']['logging_name']
    if logging_name is None:
        logging_name = "model_{}_dataset_{}_loss_{}".format(hyperparams["model"]["name"],
                                                            hyperparams["dataset"]["type"],
                                                            hyperparams["criterion"]["name"])
    shutil.rmtree("runs/{}".format(logging_name), ignore_errors=True, )
    shutil.rmtree("logs/{}".format(logging_name), ignore_errors=True, )
    os.makedirs("logs/{}".format(logging_name), exist_ok=True, )

    writer = SummaryWriter("runs/{}".format(logging_name))
    logging_dir = "logs/{}".format(logging_name)
    logging_path = os.path.join(logging_dir, "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+",
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                    (hyperparams["train"]["local_rank"], device, n_gpus, bool(hyperparams["train"]["local_rank"] != -1),
                     hyperparams["train"]["fp16"]))

    set_seed(n_gpus, seed=520)

    # Create model
    """
    if dataset == "CUB_200_2011":
        num_classes = 200
    elif .dataset == "car":
        num_classes = 196
    elif dataset == "nabirds":
        num_classes = 555
    elif dataset == "dog":
        num_classes = 120
    elif dataset == "INat2017":
        num_classes = 5089
    """

    num_classes = hyperparams["train"]["num_classes"]
    model = build_model(cfg=hyperparams, num_classes=num_classes)
    logging.info(f"===> Create model successful  ... <===")
    logging.info(f"===> Model:\t{model}  ... <===")

    if hyperparams["train"]["pretrain_weights"] is not None:
        assert os.path.exists(hyperparams["train"]["pretrain_weights"])
        logging.info(f"===> Loading model weights from {hyperparams['train']['pretrain_weights']}... <===")
        weights_dict = torch.load(hyperparams["train"]["pretrain_weights"], map_location='cpu')
        model.load_state_dict(weights_dict["model"])
    model.to(device)
    num_params = count_parameters(model)
    logging.info(f"===> Model parameters :\t {num_params} M ... <===")

    # Create transforms
    dataset = hyperparams["dataset"]["name"]
    train_transform, val_transform = build_transforms(hyperparams, dataset=dataset)
    logging.info(f"===> Create transforms successful  ... <===")
    logging.info(f"===> train transforms:\t{train_transform} <===")
    logging.info(f"===> val transforms:\t{val_transform} <===")

    # Create dataloader
    train_loader, val_loader = build_loader(hyperparams, train_transform, val_transform)
    logging.info(
        f"===>  Create train loader successful, size: {len(train_loader)} x batch size: {hyperparams['train']['batch_size']} <===")
    logging.info(
        f"===>  Create val loader successful, size: {len(val_loader)} x batch size: {hyperparams['train']['batch_size']}  <===")

    # Create optim
    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    optimizer_params = hyperparams["optimizer"]
    criterion_params = hyperparams["criterion"]
    scheduler_params = hyperparams["scheduler"]
    optim = build_optim(params_to_optimize, optimizer_params, criterion_params, scheduler_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(hyperparams["scheduler"]["params"]["t_total"]):
    #     scheduler.step()
    #     lr = optimizer.param_groups[0]["lr"]
    #     lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if hyperparams["train"]["fp16"]:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if hyperparams["train"]["local_rank"] != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Write train params
    with open(os.path.join(logging_dir, "config.yaml"), encoding='utf-8', mode='w') as f:
        try:
            yaml.dump(data=hyperparams, stream=f, allow_unicode=True)
        except Exception as e:
            print(e)

    logging.info("\n***** Train params *****")
    for k, v in hyperparams.items():
        logging.info(f"====>  {k}: {v}   <=====")

    # Train
    logging.info("\n***** Running training *****")
    logging.info("===>  Total optimization steps = %d  <===", hyperparams["scheduler"]["params"]["t_total"])
    logging.info("===>  Instantaneous batch size per GPU = %d  <===", hyperparams["train"]["batch_size"])
    logging.info("===>  Total train batch size (w. parallel, distributed & accumulation) = %d  <===",
                 hyperparams["train"]["batch_size"] * hyperparams["train"]["gradient_accumulation_steps"] * (
                     torch.distributed.get_world_size() if hyperparams["train"]["local_rank"] != -1 else 1))
    logging.info("===>  Gradient Accumulation steps = %d  <===", hyperparams["train"]["gradient_accumulation_steps"])

    model.zero_grad()
    set_seed(n_gpus, seed=520)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=hyperparams["train"]["local_rank"] not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            loss, logits = model(images,labels)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], labels.detach().cpu().numpy(), axis=0)

            if hyperparams["train"]["gradient_accumulation_steps"] > 1:
                loss = loss / hyperparams["train"]["gradient_accumulation_steps"]
            if hyperparams["train"]["fp16"]:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % hyperparams["train"]["gradient_accumulation_steps"] == 0:
                losses.update(loss.item() * hyperparams["train"]["gradient_accumulation_steps"])
                if hyperparams["train"]["fp16"]:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), hyperparams["train"]["max_grad_norm"])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["train"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (
                        global_step, hyperparams["scheduler"]["params"]["t_total"], losses.val)
                )

                if hyperparams["train"]["local_rank"] in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % hyperparams["train"]["eval_every"] == 0:
                    with torch.no_grad():
                        accuracy = valid(hyperparams, model, writer, val_loader, global_step, device)
                    if hyperparams["train"]["local_rank"] in [-1, 0]:
                        if best_acc < accuracy:
                            os.makedirs("weights/{}".format(logging_name), exist_ok=True, )
                            save_model(hyperparams, model, "weights/{}".format(logging_name))
                            best_acc = accuracy
                        logging.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % hyperparams["scheduler"]["params"]["t_total"] == 0:
                    break

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(device)
        if hyperparams["train"]["local_rank"] not in [-1, 0]:
            dist.barrier()
        train_accuracy = reduce_mean(accuracy, nprocs=hyperparams["train"]['nprocs'])
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logging.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % hyperparams["scheduler"]["params"]["t_total"] == 0:
            break

    writer.close()
    logging.info("====>  Best Accuracy: \t%f  <===" % best_acc)
    logging.info("******End Training!*****")
    end_time = time.time()
    logging.info("====>  Total Training Time: \t%f   <====" % ((end_time - start_time) / 3600))
    writer.close()


def valid(config, model, writer, test_loader, global_step, device):
    # Validation!
    eval_losses = AverageMeter()

    logging.info("***** Running Validation *****")
    logging.info("===>  Num steps = %d  <===", len(test_loader))
    logging.info("===>  Batch size = %d  <===", config["train"]["batch_size"])

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=config["train"]["local_rank"] not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(device)
    if config["train"]["local_rank"] not in [-1, 0]:
        dist.barrier()
    val_accuracy = reduce_mean(accuracy, config["train"]["nprocs"])
    val_accuracy = val_accuracy.detach().cpu().numpy()

    logging.info("\n")
    logging.info("Validation Results")
    logging.info("Global Steps: %d" % global_step)
    logging.info("Valid Loss: %2.5f" % eval_losses.avg)
    logging.info("Valid Accuracy: %2.5f" % val_accuracy)
    if config["train"]["local_rank"] in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)

    return val_accuracy


if __name__ == '__main__':
    # Usage
    """
    python train.py --config_name config/example.yaml
    """
    hyperparams = parse_config()
    run(hyperparams)
