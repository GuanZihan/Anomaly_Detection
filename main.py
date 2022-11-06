import matplotlib.pyplot as plt

from Utils.data_loader import *
from config import *


def display_image(dataloader):
    """
    Display the first 8 images in a dataloader
    :param dataloader: dataloader to be displayed
    :return:
    """
    # create figure
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 4

    for idx, (img, target) in enumerate(dataloader):
        for i in range(8):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img[i].permute(1, 2, 0))
        break
    plt.show()


if __name__ == '__main__':
    """
    Generating and display anomaly images
    """
    opt = get_arguments().parse_args()
    train_bad_loader, bad_subset_loader, clean_subset_loader, train_data_bad, perm = get_anomaly_loader(opt)

    display_image(train_bad_loader)
    display_image(clean_subset_loader)
    display_image(bad_subset_loader)



