from models.selector import *
from Utils.util import *
from Utils.data_loader import *
from torch.utils.data import DataLoader
from config import get_arguments
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def compute_loss_value(opt, poisoned_data, model_ascent):
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record_original = []
    example_data_loader = DataLoader(dataset=poisoned_data,
                                     batch_size=1,
                                     shuffle=False,
                                     )
    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
        losses_record_original.append(loss.item())

    losses_idx = np.argsort(np.array(losses_record_original))
    return losses_idx


def isolate_data(poisoned_data, losses_idx, ratio):
    # Initialize lists
    other_examples = []
    isolation_examples = []

    cnt = 0

    example_data_loader = DataLoader(dataset=poisoned_data,
                                     batch_size=1,
                                     shuffle=False,
                                     )

    perm = losses_idx[0: int(len(losses_idx) * ratio)]

    for idx, (img, target) in tqdm(enumerate(example_data_loader, start=0)):
        img = img.squeeze()
        target = target.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        # Filter the examples corresponding to losses_idx
        if idx in perm:
            isolation_examples.append((img, target))
        else:
            other_examples.append((img, target))

    logger.info('Finish collecting {} isolation examples: '.format(len(isolation_examples)))
    logger.info('Finish collecting {} other examples: '.format(len(other_examples)))
    return set(perm)


def train_step(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss_ascent = criterion(output, target)

        optimizer.zero_grad()
        loss_ascent.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_ascent.item(), img.size(0))

        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        if idx % opt.print_freq == 0:
            logger.info('Epoch[{0}]:[{1:03}/{2:03}] '
                        'Loss:{losses.val:.4f}({losses.avg:.4f})  '
                        'Prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                        'Prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                       top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch, mode):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()
    # idx batch idx
    for idx, (img, target) in enumerate(test_clean_loader, start=1):

        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    logger.info('[Dataset] {} [Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(mode, acc_clean[0], acc_clean[2]))
    logger.info('[Dataset] {} [Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(mode, acc_bd[0], acc_bd[2]))

    return acc_clean, acc_bd


def train(opt):
    # intialize two sets for storing
    # perm1: true permutation_index
    # perm2: isolated permutation_index
    # Load models
    logger.info('----------- Network Initialization --------------')
    model_ascent, _ = select_model(dataset=opt.dataset,
                                   model_name=opt.model_name,
                                   pretrained=False,
                                   pretrained_models_path="",
                                   n_classes=opt.num_class)
    if opt.cuda:
        model_ascent.to(opt.device)

    logger.info('finished model init...')

    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    log_screen = True
    handler = logging.FileHandler(
        "./logs/Injection_ratio{}.txt".format(opt.inject_portion),
        encoding='utf-8', mode='a')
    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(handler)

    logger.info('----------- Data Initialization --------------')
    train_bad_loader, bad_subset_loader, clean_subset_loader, train_data_bad, perm = get_anomaly_loader(opt)
    np.save("perm_index.npy", np.array(perm))
    perm1 = set(perm)

    test_clean_loader_testset, test_bad_loader_testset = get_test_loader(opt)

    test_bad_loader_trainset = bad_subset_loader
    test_clean_loader_trainset = clean_subset_loader

    best_clean_testset = 0
    Ip_dict = {}
    losses_clean = []
    losses_bad = []

    logger.info('----------- Train Initialization --------------')

    for epoch in range(0, opt.tuning_epochs):

        # train every epoch
        if epoch == 0:
            acc_clean_trainset, acc_bad_trainset = test(opt, test_clean_loader_trainset, test_bad_loader_trainset,
                                                        model_ascent, criterion, epoch + 1, mode="train")
            test(opt, test_clean_loader_testset, test_bad_loader_testset, model_ascent,
                 criterion, epoch + 1, mode="test")
            losses_clean.append(acc_clean_trainset[2])
            losses_bad.append(acc_bad_trainset[2])

        train_step(opt, train_bad_loader, model_ascent, optimizer, criterion, epoch + 1)

        scheduler.step()
        logger.info('lr: {}'.format(optimizer.param_groups[0]['lr']))

        # evaluate on training set
        logger.info('testing the ascended model on training dataset......')
        logger.info("clean {}".format(len(test_clean_loader_trainset)))
        logger.info("bad {}".format(len(test_bad_loader_trainset)))
        acc_clean_trainset, acc_bad_trainset = test(opt, test_clean_loader_trainset, test_bad_loader_trainset,
                                                    model_ascent, criterion, epoch + 1, mode="train")

        # evaluate on testing set
        logger.info('testing the ascended model on testing dataset......')
        logger.info("clean {}".format(len(test_clean_loader_testset)))
        logger.info("bad ".format(len(test_bad_loader_testset)))
        acc_clean_testset, acc_bad_testset = test(opt, test_clean_loader_testset, test_bad_loader_testset, model_ascent,
                                                  criterion, epoch + 1, mode="test")

        losses_clean.append(acc_clean_trainset[2])
        losses_bad.append(acc_bad_trainset[2])

        # compute the loss value and isolate data at iteration 20
        # if epoch < 10:
        #     logger.info('----------- Calculate loss value per example -----------')
        #     losses_idx = compute_loss_value(opt, train_data_bad, model_ascent)
        #     logger.info('----------- Collect isolation data -----------')
        #     IPs = []
        #     for i in [opt.inject_portion]:
        #         perm2 = isolate_data(train_data_bad, losses_idx, i)
        #         intersection = perm1 & perm2
        #         ip = len(intersection) / len(perm2)
        #         IPs.append(ip)
        #         print(
        #             "[Attack] {} [isolation_ratio] {:.2f} [Isolation Precision] {:.2f}".format(opt.trigger_type, i, ip))
        #     Ip_dict[epoch] = IPs
    np.save("losses_bad.npy", losses_bad)
    np.save("losses_clean.npy", losses_clean)

    return train_data_bad, model_ascent, best_clean_testset, Ip_dict


def main():
    opt = get_arguments().parse_args()
    logger.info('attack: ' + str(opt.trigger_type) + " injection ratio: " + str(opt.inject_portion))
    train(opt)


if __name__ == '__main__':
    main()
