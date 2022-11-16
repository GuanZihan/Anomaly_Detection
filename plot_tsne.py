import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt


# backdoor_heatmap_original
from config import get_arguments


def visualize_tsne_points(tx, ty, labels, perm):
    print('Plotting TSNE image')
    # initialize matplotlib plot
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)

    class_name = ['good', 'missing', 'shift', 'stand', 'broke', 'short']

    colors_per_class = {
        0: [254, 202, 87],
        1: [255, 107, 107],
        2: [10, 189, 227],
        3: [255, 159, 243],
        4: [16, 172, 132],
        5: [128, 80, 128],
        6: [33, 20, 15],
        7: [180, 20, 25],
        8: [10, 150, 25],
        9: [180, 20, 66],
    }
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        if (label == 3):
            indices_good = []
            indices_bad = []
            for i, l in enumerate(labels):
                if l == label and i in perm:
                    indices_bad.append(i)
                if l == label and i not in perm:
                    indices_good.append(i)
            current_tx = np.take(tx, indices_good)
            current_ty = np.take(ty, indices_good)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label

            ax.scatter(current_tx, current_ty, c=color, label=label)

            current_tx = np.take(tx, indices_bad)
            current_ty = np.take(ty, indices_bad)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([[0, 0, 255][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label

            ax.scatter(current_tx, current_ty, c=color, label="3-poisoned")


        else:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label

            ax.scatter(current_tx, current_ty, c=color, label=label)

            # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()
    plt.savefig('visualize_tsne_points.png')


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne(tsne, labels, perm, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, perm)

    # visualize the plot: samples as images
    # visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


opt = get_arguments().parse_args()

features = np.load("./experiments/features_{}.npy".format(opt.trigger_type), allow_pickle=True)[:10000]
labels = np.load("./experiments/labels_{}.npy".format(opt.trigger_type), allow_pickle=True)[:10000]
perm1 = np.load("perm_index.npy", allow_pickle=True)

print(features.shape)
tsne = manifold.TSNE(n_components=2).fit_transform(features)
visualize_tsne(tsne, labels, perm1)
