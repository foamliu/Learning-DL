import pandas as pd
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


# Define a function to show image through 48*48 pixels
def show(img):
    show_image = img.reshape(48, 48)

    # plt.imshow(show_image, cmap=cm.binary)
    plt.imshow(show_image, cmap='gray')
    plt.show()


def read_data(file):
    data = pd.read_csv(file)
    print(data.shape)
    print(data.head())
    print(np.unique(data["Usage"].values.ravel()))
    print ( 'The number of data set is %d' %(len(data)))
    pixels_values = data.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.float)
    print(images)

    # show one image
    show(images[8])

    print('images.shape: ' + str(images.shape))

    image_pixels = images.shape[1]
    print( 'Flat pixel values is %d' %(image_pixels))
    image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)
    print('image_width: ' + str(image_width))
    print('image_height: ' + str(image_height))
    labels_flat = data["emotion"].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    print(np.unique(labels_flat))
    print( 'The number of different facial expressions is %d' %labels_count)
    print(labels_flat)
    print('labels.shape: ' + str(labels_flat.shape))
    print ( 'The number of final data: %d' %(len(images)))

    return images, labels_flat


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_tsne, title=None):
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 7.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X_tsne.shape[0]):
    #         dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X_tsne[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(images[i].reshape(image_width,image_height), cmap=plt.cm.gray),
    #             X_tsne[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    image_pixels = 2304
    labels_count = 7
    image_width = 48
    image_height = 48
    file = '../fer2013/fer2013.csv'
    label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    import pickle
    import os.path

    images, labels = read_data(file)

    fname = '/Users/yangliu/code/deep_learning/tools/data.pkl'
    if os.path.isfile(fname):
        with open(fname, 'rb') as input:
            X_tsne = pickle.load(input)

    else:

        X = images

        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

        X_tsne = tsne.fit_transform(X)

        with open(fname, 'wb') as output:
            pickle.dump(X_tsne, output)

    t0 = time()
    plot_embedding(X_tsne[0:300],
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))

    plt.show()
