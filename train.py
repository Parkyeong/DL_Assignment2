import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import ResNet18, ResNet34, ResNet50
from model.vgg import VGG16, VGG19

from utils import evaluate_standard, get_loaders

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--num_grids', default=1, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', type=str, default='0')  # gpu
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def setup_logging(save_dir):
    logfile = os.path.join(save_dir, 'output.log')
    
    # Check if the log file already exists
    if os.path.exists(logfile):
        with open(logfile, 'a') as f:
            # Add the time restart marker
            restart_time = time.strftime("%H:%M:%S %d/%m/%Y")
            f.write(f"\nResume from previous work, time restart from {restart_time}\n")
    
    # Set up logging with append mode
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    
def main():
    args = get_args()
        
    # if args.debug_mode:
    #     print("Debug mode enabled")
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    # saving path
    path = os.path.join('ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    # logger
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logfile = os.path.join(args.save_dir, 'output.log')
    
            
    if os.path.exists(logfile):
        with open(logfile, 'a') as f:
            restart_time = time.strftime("%H:%M:%S %d/%m/%Y")
            f.write(f"\nResume from previous work, time restart from {restart_time}\n")
    
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    # set current device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    ngpus = torch.cuda.device_count()
    logger.info('Devices: [{:d}]'.format(ngpus))

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker)

    # setup network
    if args.network == 'ResNet18':
        net = ResNet18
    elif args.network == 'ResNet34':
        net = ResNet34
    elif args.network == 'ResNet50':
        net = ResNet50
    elif args.network == 'VGG16':
        net = VGG16
    elif args.network == 'VGG19':
        net = VGG19

    else:
        print('Wrong network:', args.network)
        exit()

    model = net(num_classes=args.num_classes).cuda()
    model = torch.nn.DataParallel(model)
    logger.info(model)

    # set weight decay
    params = [{'params': param} for name, param in model.named_parameters()]

    # setup optimizer, loss function, LR scheduler
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if args.lr_schedule == 'cyclic':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, lr_steps * 3 // 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=float(args.epochs))

    # load pretrained model
    if os.path.exists(os.path.join(args.save_dir, 'model_latest.pth')):
        pretrained_model = torch.load(os.path.join(args.save_dir, 'model_latest.pth'), map_location='cpu')
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
        try:
            opt.load_state_dict(pretrained_model['opt'])
        except:
            print("Cannot load optimizer state")
        scheduler.load_state_dict(pretrained_model['scheduler'])
        start_epoch = pretrained_model['epoch'] + 1
        best_acc = pretrained_model['best_acc']
        print('Resume from Epoch %d. Loaded pretrained weights.' % start_epoch)
    else:
        start_epoch = 0
        best_acc = 0
        print('No checkpoint. Training from scratch.')

    # Start training
    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to('cuda'), y.to('cuda')

            output = model(X)
            loss = criterion(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if i % 150 == 0:
                logger.info("Device: [{:s}]\t"
                            "Iter: [{:d}][{:d}/{:d}]\t"
                            "Loss {:.3f} ({:.3f})\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                    args.device,
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    train_loss / train_n,
                    (output.max(1)[1] == y).sum().item() / y.size(0),
                    train_acc / train_n)
                )

        # Evaluate on test set
        test_loss, test_acc = evaluate_standard(test_loader, model)
        logger.info('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(test_loss, test_acc))

        # Save the model if it has the best accuracy so far
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.save_dir, 'model_best.pth'))
            logger.info('Model saved at epoch {} with best accuracy {:.4f}'.format(epoch, best_acc))

        # Save the latest model at the end of each epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'opt': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
        }, os.path.join(args.save_dir, 'model_latest.pth'))

    logger.info('Training completed in {:.2f} minutes.'.format((time.time() - start_train_time) / 60))


if __name__ == '__main__':
    main()
