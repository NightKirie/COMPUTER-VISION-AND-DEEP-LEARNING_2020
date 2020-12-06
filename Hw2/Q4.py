import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA


def Image_Reconstruction():
    imgs = os.listdir("./Q4_Image")
    plt_width = int(len(imgs) / 2)
    for i in range(plt_width):
        img = cv2.imread("./Q4_Image/" + imgs[i])
        
        plt.subplot(4, plt_width, i+1)
        plt.imshow(img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 0:
            plt.ylabel('original')

        b_img = img[:, :, 0]
        g_img = img[:, :, 1]
        r_img = img[:, :, 2]

        pca_b = PCA(75)
        pca_g = PCA(75)
        pca_r = PCA(75)

        pca_b_img = pca_b.fit_transform(b_img)
        pca_g_img = pca_g.fit_transform(g_img)
        pca_r_img = pca_r.fit_transform(r_img)
        
        approx_b_img = pca_b.inverse_transform(pca_b_img)
        approx_g_img = pca_g.inverse_transform(pca_g_img)
        approx_r_img = pca_r.inverse_transform(pca_r_img)
        
        pca_img = np.stack([approx_b_img, approx_g_img, approx_r_img], axis=2).astype("uint8")

        plt.subplot(4, plt_width, plt_width+i+1)
        plt.imshow(pca_img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == 0:
            plt.ylabel('reconstruction')

    for i in range(plt_width, len(imgs)):
        img = cv2.imread("./Q4_Image/" + imgs[i])

        plt.subplot(4, plt_width, plt_width+i+1)
        plt.imshow(img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == plt_width:
            plt.ylabel('original')

        b_img = img[:, :, 0]
        g_img = img[:, :, 1]
        r_img = img[:, :, 2]

        pca_b = PCA(75)
        pca_g = PCA(75)
        pca_r = PCA(75)

        pca_b_img = pca_b.fit_transform(b_img)
        pca_g_img = pca_g.fit_transform(g_img)
        pca_r_img = pca_r.fit_transform(r_img)
        
        approx_b_img = pca_b.inverse_transform(pca_b_img)
        approx_g_img = pca_g.inverse_transform(pca_g_img)
        approx_r_img = pca_r.inverse_transform(pca_r_img)
        
        pca_img = np.stack([approx_b_img, approx_g_img, approx_r_img], axis=2).astype("uint8")

        plt.subplot(4, plt_width, plt_width*2+i+1)
        plt.imshow(pca_img)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if i == plt_width:
            plt.ylabel('reconstruction')
    plt.show()

def Compile_Error():
    imgs = os.listdir("./Q4_Image")
    for i in range(len(imgs)):
        img = cv2.imread("./Q4_Image/" + imgs[i], cv2.IMREAD_GRAYSCALE)

        pca = PCA(75)

        pca_img = pca.fit_transform(img)
        
        approx_img = pca.inverse_transform(pca_img).astype('uint8').astype('int32')
        
        total_loss = 0
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                total_loss += abs(img[i][j] - approx_img[i][j])
        print(total_loss, end=", ")

    print('')

if __name__ == "__main__":
    Compile_Error()