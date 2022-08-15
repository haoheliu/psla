# Yuan Gong, modified from:
# Author: David Harwath
import argparse
from inspect import ArgSpec
import os
import pickle
import sys
from collections import OrderedDict
import time
import torch
import shutil
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloaders
from utilities import *
import models
from models.neural_sampler import *
from traintest import train, validate
import ast
from torch.utils.data import WeightedRandomSampler
import numpy as np
import logging
import wandb
import torch.distributed as dist
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing as mp
from utilities.sampler import DistributedSamplerWrapper
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def seed_torch(seed=1029):
    print("Set seed to %s" % seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def adaptive_batchsize(args):
    args.batch_size = int(22 * (args.target_length/1056) / args.preserve_ratio)
    return args

def main():
    print(os.getcwd())
    # I/O args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-train", type=str, default='', help="training data json")
    parser.add_argument("--data-val", type=str, default='', help="validation data json")
    parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default=os.path.join(basepath, 'utilities/class_labels_indices_coarse.csv'), help="csv with class labels")
    parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")

    # training and optimization args
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=60, type=int, metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
    parser.add_argument("--n-print-steps", type=int, default=1, help="number of steps to print statistics")

    # model args
    parser.add_argument("--model", type=str, default="efficientnet", help="audio model architecture", choices=["efficientnet", "resnet", "mbnet"])
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands","fsd50k","audiosetbalanced"])
    parser.add_argument("--graph_weight_path", type=str, default="")

    parser.add_argument("--dataset_mean", type=float, default=-4.6476, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", type=float, default=4.5699, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", type=int, default=1056, help="the input length in frames")
    parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)
    parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics", choices=["mAP", "acc"])
    parser.add_argument("--warmup", help='if use balance sampling', type=ast.literal_eval)
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function", choices=["BCE", "CE"])
    parser.add_argument("--lrscheduler_start", type=int, default=10, help="when to start decay")
    parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay ratio")
    parser.add_argument("--wa", help='if do weight averaging', type=ast.literal_eval)
    parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging")
    parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging")

    parser.add_argument("--n_class", type=int, default=527, help="number of classes")
    parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
    parser.add_argument("--eff_b", type=int, default=0, help="which efficientnet to use, the larger number, the more complex")
    parser.add_argument('--esc', help='If doing an ESC exp, which will have some different behabvior', type=ast.literal_eval, default='False')
    parser.add_argument('--impretrain', help='if use imagenet pretrained CNNs', type=ast.literal_eval, default='True')
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
    parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
    parser.add_argument('--bal', help='if use balance sampling', type=ast.literal_eval)

    parser.add_argument("--sampler", type=str, default="NeuralSampler")
    parser.add_argument("--weight_func", type=str, default="")
    parser.add_argument("--note", type=str, default="debug")
    parser.add_argument("--preserve_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0, help="The scaling factor to the importance score")
    parser.add_argument("--beta", type=float, default=1.0, help="The scaling factor to the graph weight")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reweight_loss", type=ast.literal_eval, default=False)

    args = parser.parse_args()
    
    seed_everything(int(args.seed)) # TODO put it where?
    seed_torch(int(args.seed)) # TODO put it where?
    
    """Assume Single Node Multi GPUs Training Only"""
    n_gpus = torch.cuda.device_count()
    # args.batch_size=args.batch_size*n_gpus
    args = adaptive_batchsize(args)
    if  n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args,),join=True)
    else:
        run(0, 1, args)

def run(rank, n_gpus, args):
    
    if(rank == 0):
        wandb.init(
        project="iclr2023",
        # mode="disabled", # TODO
        name=os.path.basename(args.exp_dir),
        notes=args.note,
        tags=[args.sampler],
        config=vars(args),
        )
    
    print(f"Running DDP on rank {rank}.")
    if(rank == 0):
        print("Logging directory:", args.exp_dir)
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename="%s/log.txt" % args.exp_dir,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(process)d: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        
        logging.info("GPU counts %s" % n_gpus)
        
    if(n_gpus > 1):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '60000'
        dist.init_process_group("gloo", rank=rank, world_size=n_gpus)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    
    torch.manual_seed(args.seed) # TODO set seed again
    torch.cuda.set_device(rank)
    
    g = torch.Generator()
    g.manual_seed(0)

    logging.info("I am process %s, running on %s: starting (%s)" % (
            os.getpid(), os.uname()[1], time.asctime()))
    print("I am process %s, running on %s: starting (%s)" % (
            os.getpid(), os.uname()[1], time.asctime()))

    audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm,
                'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train',
                'mean': args.dataset_mean, 'std': args.dataset_std,
                'noise': False}
    
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                    'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                    'std': args.dataset_std, 'noise': False}

    if args.bal == True:
        logging.info('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        
        if(n_gpus>1):
            sampler = DistributedSamplerWrapper(sampler, num_replicas=n_gpus, rank=rank, shuffle=True)
            
        dataset = dataloaders.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True, worker_init_fn=seed_worker,generator=g)
        logging.info("The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s" % (len(dataset), len(train_loader), args.batch_size))
    else:
        logging.info('balanced sampler is not used')
        dataset = dataloaders.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
        
        if(n_gpus > 1):
            sampler = DistributedSampler(dataset, num_replicas=n_gpus, rank=rank, shuffle=False)
        else:
            sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True, worker_init_fn=seed_worker,generator=g)
        
        logging.info("The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s" % (len(dataset), len(train_loader), args.batch_size))

    if(rank==0):
        val_loader = torch.utils.data.DataLoader(
            dataloaders.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker,generator=g)

        if args.data_eval != None:
            eval_loader = torch.utils.data.DataLoader(
                dataloaders.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
                batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker,generator=g)

    if args.model == 'efficientnet':
        audio_model = models.EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain, head_num=args.att_head, input_seq_length=args.target_length,sampler=eval(args.sampler), preserve_ratio=args.preserve_ratio, alpha=args.alpha)
    elif args.model == 'resnet':
        audio_model = models.ResNetAttention(label_dim=args.n_class, pretrain=args.impretrain)
    elif args.model == 'mbnet':
        audio_model = models.MBNet(label_dim=args.n_class, pretrain=args.effpretrain)
    audio_model = audio_model.cuda(rank)
    print("===> Woking directory:", os.getcwd())
    
    audio_model.rank = rank
    
    if(n_gpus > 1):
        audio_model = DDP(audio_model, device_ids=[rank])
    
    # if you want to use a pretrained model for fine-tuning, uncomment here.
    if("audioset" not in args.dataset):
        # TODO we might have error here
        print("Reloading model trained on audioset, mAP 0.4329")
        logging.info("Reloading model trained on audioset, mAP 0.4329")
        device = torch.device("cuda:%s" % rank if torch.cuda.is_available() else "cpu")
        sd = torch.load('../../pretrained_models/as_mdl_0_wa.pth', map_location=device)
        audio_model.module.load_state_dict(sd, strict=False)

    if(rank==0):
        logging.info("\nCreating experiment directory: %s" % args.exp_dir)
        if os.path.exists("%s/models" % args.exp_dir) == False:
            os.makedirs("%s/models" % args.exp_dir)
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)
        logging.info("Initializing...")
        
    if(rank == 0):
        train(rank, n_gpus, audio_model, train_loader, val_loader, args)
    else:
        train(rank, n_gpus, audio_model, train_loader, None, args)

    # if the dataset has a seperate evaluation set (e.g., FSD50K), then select the model using the validation set and eval on the evaluation set.
    logging.info('---------------Result Summary---------------')
    info = {}
    if args.data_eval != None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # evaluate best single model
        sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd)
        logging.info('---------------evaluate best single model on the validation set---------------')
        stats, _ = validate(audio_model, val_loader, args, 'best_single_valid_set')
        val_mAP = np.mean([stat['AP'] for stat in stats])
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        logging.info("mAP: {:.6f}".format(val_mAP))
        logging.info("AUC: {:.6f}".format(val_mAUC))
        info["mAP/val_single"]=val_mAP
        info["AUC/val_single"]=val_mAUC
        logging.info('---------------evaluate best single model on the evaluation set---------------')
        stats, _ = validate(audio_model, eval_loader, args, 'best_single_eval_set', eval_target=True)
        eval_mAP = np.mean([stat['AP'] for stat in stats])
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        logging.info("mAP: {:.6f}".format(eval_mAP))
        logging.info("AUC: {:.6f}".format(eval_mAUC))
        info["mAP/eval_single"]=eval_mAP
        info["AUC/eval_single"]=eval_mAUC
        np.savetxt(args.exp_dir + '/best_single_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

        # evaluate weight average model
        sd = torch.load(args.exp_dir + '/models/audio_model_wa.pth', map_location=device)
        audio_model.load_state_dict(sd)
        logging.info('---------------evaluate weight average model on the validation set---------------')
        stats, _ = validate(audio_model, val_loader, args, 'wa_valid_set')
        val_mAP = np.mean([stat['AP'] for stat in stats])
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        logging.info("mAP: {:.6f}".format(val_mAP))
        logging.info("AUC: {:.6f}".format(val_mAUC))
        info["mAP/val_wa"]=val_mAP
        info["AUC/val_wa"]=val_mAUC
        logging.info('---------------evaluate weight averages model on the evaluation set---------------')
        stats, _ = validate(audio_model, eval_loader, args, 'wa_eval_set')
        eval_mAP = np.mean([stat['AP'] for stat in stats])
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        logging.info("mAP: {:.6f}".format(eval_mAP))
        logging.info("AUC: {:.6f}".format(eval_mAUC))
        info["mAP/eval_wa"]=eval_mAP
        info["AUC/eval_wa"]=eval_mAUC
        np.savetxt(args.exp_dir + '/wa_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

        # evaluate the ensemble results
        logging.info('---------------evaluate ensemble model on the validation set---------------')
        # this is already done in the training process, only need to load
        result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
        val_mAP = result[-1, -3]
        val_mAUC = result[-1, -2]
        logging.info("mAP: {:.6f}".format(val_mAP))
        logging.info("AUC: {:.6f}".format(val_mAUC))
        info["mAP/val_ensemble"]=val_mAP
        info["AUC/val_ensemble"]=val_mAUC
        logging.info('---------------evaluate ensemble model on the evaluation set---------------')
        # get the prediction of each checkpoint model
        for epoch in range(1, args.n_epochs+1):
            sd = torch.load(args.exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
            audio_model.load_state_dict(sd)
            validate(audio_model, eval_loader, args, 'eval_'+str(epoch))
        # average the checkpoint prediction and calculate the results
        target = np.loadtxt(args.exp_dir + '/predictions/eval_target.csv', delimiter=',')
        ensemble_predictions = np.zeros_like(target)
        for epoch in range(1, args.n_epochs + 1):
            cur_pred = np.loadtxt(args.exp_dir + '/predictions/predictions_eval_' + str(epoch) + '.csv', delimiter=',')
            ensemble_predictions += cur_pred
        ensemble_predictions = ensemble_predictions / args.n_epochs
        stats = calculate_stats(ensemble_predictions, target, args)
        eval_mAP = np.mean([stat['AP'] for stat in stats])
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        logging.info("mAP: {:.6f}".format(eval_mAP))
        logging.info("AUC: {:.6f}".format(eval_mAUC))
        info["mAP/eval_ensemble"]=eval_mAP
        info["AUC/eval_ensemble"]=eval_mAUC
        np.savetxt(args.exp_dir + '/ensemble_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

    # if the dataset only has evaluation set (no validation set), e.g., AudioSet
    else:
        # evaluate single model
        logging.info('---------------evaluate best single model on the evaluation set---------------')
        # result is the performance of each epoch, we average the results of the last 5 epochs
        result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
        last_five_epoch_mean = np.max(result[-20: , :], axis=0)
        eval_mAP = last_five_epoch_mean[0]
        eval_mAUC = last_five_epoch_mean[1]
        logging.info("mAP: {:.6f}".format(eval_mAP))
        logging.info("AUC: {:.6f}".format(eval_mAUC))
        info["mAP/2_eval_single"]=eval_mAP
        info["AUC/2_eval_single"]=eval_mAUC
        np.savetxt(args.exp_dir + '/best_single_result.csv', [eval_mAP, eval_mAUC])

        # evaluate weight average model
        logging.info('---------------evaluate weight average model on the evaluation set---------------')
        # already done in training process, only need to load
        result = np.loadtxt(args.exp_dir + '/wa_result.csv', delimiter=',')
        wa_mAP = result[0]
        wa_mAUC = result[1]
        logging.info("mAP: {:.6f}".format(wa_mAP))
        logging.info("AUC: {:.6f}".format(wa_mAUC))
        info["mAP/2_eval_wa"]=wa_mAP
        info["AUC/2_eval_wa"]=wa_mAUC
        np.savetxt(args.exp_dir + '/wa_result.csv', [wa_mAP, wa_mAUC])

        # evaluate ensemble
        logging.info('---------------evaluate ensemble model on the evaluation set---------------')
        # already done in training process, only need to load
        result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
        ensemble_mAP = result[-1, -3]
        ensemble_mAUC = result[-1, -2]
        logging.info("mAP: {:.6f}".format(ensemble_mAP))
        logging.info("AUC: {:.6f}".format(ensemble_mAUC))
        info["mAP/2_eval_ensemble"]=ensemble_mAP
        info["AUC/2_eval_ensemble"]=ensemble_mAUC
        np.savetxt(args.exp_dir + '/ensemble_result.csv', [ensemble_mAP, ensemble_mAUC])
    
    if(rank == 0):
        # Log the result
        wandb.log_artifact(os.path.join(args.exp_dir,"log.txt"), name='logging_file', type='txt') 
        for k in info.keys(): info[k] = float(info[k])
        print(info)
        wandb.log(info)
        wandb.finish()



if __name__ == "__main__":
    main()
    cleanup()