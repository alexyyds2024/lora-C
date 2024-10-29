import torch
import torchvision
from torch import optim, nn
from torch.utils import data
from torchvision.transforms import transforms
import numpy as np
import loralib as lora
from model import resnet_lora
import argparse
import os
import torchvision.datasets as datasets
import logging
import time
from utils import custom_cifar


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train(net, epoch, trainloader, logger):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            logger.info('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if torch.isnan(torch.tensor(loss.item())):
            return True

    return False


def test(net, testloader, logger):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    correct_sample_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct_sample_indices.extend(predicted.eq(targets).cpu().detach().numpy().tolist())
            correct += predicted.eq(targets).sum().item()
    if 100. * correct / total > best_acc:
        best_acc = 100. * correct / total
        torch.save(net.state_dict(),
                   '{}.pth'.format(checkpoint_dir))
        # cifar10_c(net)
    logger.info('Test Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total, np.where(correct_sample_indices)[0]


def add_weight_decay(model, weight_decay=5e-4, skip_list=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='choose model,eg resnet18')
    parser.add_argument('--num_classes', default=10, type=int, help='dataset number classes')
    parser.add_argument('--dataset', default='cifar10', type=str, help='choose dataset')
    parser.add_argument('--dataset_dir', default='./datasets/cifar10', type=str, help='dataset dir')
    parser.add_argument('--checkpoint_base_dir', default='./lora-C', type=str,
                        help='checkpoint base dir')
    parser.add_argument('--test_style', type=str, default='None', help='Icons-50 test style')
    parser.add_argument('--log_dir', type=str, default='./lora-C', help='log storage path')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--ratio', type=float, default=1.0, help='Select the ratio of the dataset')
    parser.add_argument('--epoch', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--load_pth', type=str,
                        default='./lora-C/weight/resnet18-f37072fd.pth', help='Pre-trained weight path')
    parser.add_argument('--device', default='1', type=str, help='device')
    parser.add_argument('--rank', default=16, type=int, help='lora r')
    parser.add_argument('--alpha', default=32, type=int, help='lora alpha')
    parser.add_argument('--lora_dropout', default=0., type=float, help='lora dropout')

    args = parser.parse_args()

    lora_config = {
        'r': args.rank,
        'lora_alpha': args.alpha,
        'lora_dropout': args.lora_dropout,
        'merge_weights': True
    }
    logger_dir = args.log_dir
    logger_path = '{}/logger_{}/train_{}_{}_rank_{:d}_alpha_{:d}_{}_{}'.format(args.log_dir, args.model, args.model,
                                                                               args.dataset,
                                                                               args.rank, args.alpha,
                                                                               args.load_weight_tag,
                                                                               args.ratio)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    checkpoint_dir = '{}/checkpoint_{}/{}_{}_rank_{:d}_alpha_{:d}_{}'.format(args.checkpoint_base_dir, args.model,
                                                                             args.model, args.dataset,
                                                                             args.rank, args.alpha,
                                                                             args.load_weight_tag)

    if args.dataset == 'icons50':
        checkpoint_dir = '{}_{}'.format(checkpoint_dir, args.test_style)
    if args.model == 'resnet18':
        model = resnet_lora.ResNet18(True, lora_config, num_classes=args.num_classes)
        state = torch.load(args.load_pth)
    elif args.model == 'resnet34':
        model = resnet_lora.ResNet34(True, lora_config, num_classes=args.num_classes)
        state = torch.load(args.load_pth)
    elif args.model == 'resnet50':
        model = resnet_lora.ResNet50(True, lora_config, num_classes=args.num_classes)
        state = torch.load(args.load_pth)
    elif args.model == 'resnet101':
        model = resnet_lora.ResNet101(True, lora_config, num_classes=args.num_classes)
        state = torch.load(args.load_pth)
    else:
        raise NotImplementedError

    model.load_weight(state)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，保留 32x32 的图像，并在每边填充 4 个像素
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        trainset = custom_cifar.CustomCIFAR10(
            root=args.dataset_dir, train=True, download=True, transform=transform_train, ratio=args.ratio)
        testset = custom_cifar.CustomCIFAR10(
            root=args.dataset_dir, train=False, download=True, transform=transform_test, ratio=args.ratio)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir, train=False, download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'icons50':
        logger_path = '{}_{}.log'.format(logger_path, args.test_style)
        train_data = datasets.ImageFolder(args.dataset_dir,
                                          transform=transforms.Compose(
                                              [transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(),
                                               transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                               # RandomErasing()
                                               ]))
        test_data = datasets.ImageFolder(args.dataset_dir,
                                         transform=transforms.Compose(
                                             [transforms.Resize((32, 32)), transforms.ToTensor()]))
        filtered_imgs = []
        for img in train_data.samples:
            img_name = img[0]
            if args.test_style not in img_name:
                filtered_imgs.append(img)

        train_data.samples = filtered_imgs[:]

        filtered_imgs = []
        for img in test_data.samples:
            img_name = img[0]
            if args.test_style in img_name:
                filtered_imgs.append(img)

        test_data.samples = filtered_imgs[:]

        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True)

    LOGGER = get_logger(logger_path)
    LOGGER.info(lora_config)

    lora_param_count = sum(p.numel() for p in model.parameters())
    LOGGER.info(lora_param_count / 1e6)

    lora.mark_only_lora_as_trainable(model)
    for n, p in model.named_parameters():
        if 'fc.weight' == n:
            p.requires_grad = True
        elif 'conv1.weight' == n:
            p.requires_grad = True

    lora_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info('parameters counts:{:f}'.format(lora_param_count / 1e6))
    for n, p in model.named_parameters():
        if p.requires_grad:
            LOGGER.info(n)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    criterion = nn.CrossEntropyLoss()

    parameters = add_weight_decay(model, 5e-4,
                                  skip_list=('linear',))  # this will remove BN parameters from weight decay
    weight_decay = 0.  # override the weight decay value

    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=0.9,
                          weight_decay=weight_decay
                          )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    model = model.to(device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for epoch in range(0, args.epoch):
        terminate = train(model, epoch, trainloader, LOGGER)
        acc, sampler_indices = test(model, testloader, LOGGER)
        scheduler.step()
        if terminate:
            break
    LOGGER.info('best acc {}'.format(best_acc))
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    LOGGER.info('Stop Training in epoch {}, Training Time is {} sec'.format(args.epoch - 1, t2 - t1))
