import cv2
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import numpy as np
import time
from tqdm import tqdm


def get_test_loader(opt):
    print('==> Preparing test data..')
    tf_test = transforms.Compose([transforms.ToTensor()
                                  ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    elif (opt.dataset == "Cifar100"):
        testset = datasets.CIFAR100(root='data/Cifar100', train=False, download=True)
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     np.array([125.3, 123.0, 113.9]) / 255.0,
            #     np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=opt.batch_size,
                                   shuffle=False,
                                   )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 )

    return test_clean_loader, test_bad_loader


def get_anomaly_loader(opt):
    """
    Return the dataloder for anomaly dataset (Type 1)
    :param opt: predefined arguments
    :return: train_data_bad: dataset for anomaly training dataset
    train_bad_loader: dataloader for anomaly training dataset
    train_data_bad.perm: the index of the anomaly data points
    bad_subset, clean_subset: the dataloader for anomaly subset, and clean subset
    """
    print('==> Preparing train data..')
    tf_train = transforms.Compose([transforms.ToTensor(),
                                   ])
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif (opt.dataset == "Cifar100"):
        trainset = datasets.CIFAR100(root='data/Cifar100', train=True, download=True)
        tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train,
                               mode='train')

    train_bad_loader = DataLoader(dataset=train_data_bad,
                                  batch_size=opt.batch_size,
                                  shuffle=False, )
    '''
    Format of train_data_bad.bad_data and train_data_bad.clean_data:
    [(img, label), (img, label), ..., (img, label)]
    img might be a 32*32*3 array, and label might be a number
    '''
    bad_subset = DataLoader(dataset=Dataset_From_List(train_data_bad.bad_data, tf_train),
                            batch_size=opt.batch_size,
                            shuffle=False, )
    clean_subset = DataLoader(dataset=Dataset_From_List(train_data_bad.clean_data, tf_train),
                              batch_size=opt.batch_size,
                              shuffle=False, )

    return train_bad_loader, bad_subset, clean_subset, train_data_bad, train_data_bad.perm


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen


class Dataset_From_List(Dataset):
    def __init__(self, full_dataset, transform=None, device=torch.device("cuda")):
        self.dataset = full_dataset
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"),
                 distance=1):
        self.dataset, self.perm, self.bad_data, self.clean_data = self.addTrigger(full_dataset, opt.target_label,
                                                                                  inject_portion, mode, distance,
                                                                                  opt.trig_w, opt.trig_h,
                                                                                  opt.trigger_type, opt.target_type)
        # self.dataset = self.dataset[:24000]
        # self.perm = np.sort(self.perm)[:2400]
        self.opt = opt
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + "bad Imgs")

        # random seed, used for voting method
        np.random.seed(12345)

        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # perm = np.arange(0, int(len(dataset) * inject_portion))
        # input()
        perm_ = perm
        if target_type == 'cleanLabel':
            perm_ = []

        dataset_ = list()
        bad_data = list()
        clean_data = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        # change target

                        dataset_.append((img, target_label))
                        bad_data.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                        clean_data.append((img, data[1]))

                else:
                    # if the label of the image is already equal to the target, then omit
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])

                    width = img.shape[0]
                    height = img.shape[1]
                    # img = np.random.randint(low=0, high=256, size=(width, height, 3), dtype=np.uint8)
                    if i in perm:

                        # img = np.random.randint(low=0, high=256, size=(width, height, 3), dtype=np.uint8)
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        # if(inject_portion == 1):
                        #   np.save("zero_img", img)
                        #   input()
                        dataset_.append((img, target_label))
                        bad_data.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                        clean_data.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))

                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':
                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1
                            perm_.append(i)


                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")
        print("After injecting ,the length of the permutated images is " + str(len(perm_)))

        return dataset_, perm_, bad_data, clean_data

    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
            # input()
        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)
        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.load("kitty.npy", allow_pickle=True)

        blend_img = (1 - alpha) * img + alpha * mask

        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2

        # load signal mask
        # signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        signal_mask = _plant_sin_trigger(img, 20, 6, False)
        # np.save("signal", signal_mask)
        # input()
        # blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        # blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return signal_mask


def _plant_sin_trigger(img, delta=60, f=4, debug=False):
    alpha = 0.8
    # img = np.float32(img)
    pattern = np.zeros_like(img)

    m = pattern.shape[1]

    if img.shape == (28, 28):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    else:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    pattern[i, j, k] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))
    # np.save("signal.npy", img)
    # input()
    return img