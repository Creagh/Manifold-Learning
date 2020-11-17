"""
**********************************************************
* Dimensionality Reduction using the Famous Iris Dataset *
**********************************************************
****************************
* COMP 4190 - Assignment 4 *
* by Creagh Briercliffe    *
****************************

Data acquired from: UCI Machine Learning Repository
http://archive.ics.uci.edu/ml/datasets/Iris
Creator: R.A. Fisher

"""

import os
import csv
import pylab as pl
import numpy as np
from matplotlib import offsetbox
from os.path import dirname, join
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn import manifold, decomposition, ensemble, lda, random_projection


n_samples = 150
n_features = 4

class Dataset(dict):
    """
    A dictionary object to hold the entire dataset, including features,
    targets, and feature names.
    
    Code borrowed from the sklearn base.py file.
    Authors: David Cournapeau, Fabian Pedregosa & Olivier Grisel
    """
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def import_data():
    """ 
    Import the Iris dataset, a CSV file, containing 4 columns of numeric
    attributes and a 5th column with class labels as strings.
    """
    path = dirname(__file__)
    file = csv.reader(open(join(os.pardir, 'data/bezdekIris.csv'), 'rU'))
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=object)

    for i, j in enumerate(file):
        data[i] = np.asarray(j[:-1], dtype=np.float)
        target[i] = np.asarray(j[-1], dtype=object)

    return Dataset(data=data, target=target, feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'])


# Import the Iris dataset
dataset = import_data()
# Seperate the dataset into features (X) and targets (Y)
X = dataset.data[:, :4]
Y = dataset.target

n_neighbors = 50
n_components = 2

# Make a list that gives each target class a unique int value
# These int values can be used to give different colors to the data points in plots
colors = []
for i in Y:
    if i == "Iris-setosa":
        colors.append(0)
    elif i == "Iris-versicolor":
        colors.append(1)
    else:
        colors.append(2)

"""
Code borrowed in part from plot_lle_digits.py
Authors: Fabian Pedregosa, Olivier Grisel, Mathieu Blondel & Gael Varoquaux
http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#example-manifold-plot-lle-digits-py
"""
def plot_embedding(X, title=None):
    """
    Scale and plot the embedding feature vectors
    """
    # Scale the feature vectors
    x_min = np.min(X,0)
    x_max = np.max(X,0)
    X = (X - x_min) / (x_max - x_min)

    pl.figure()
    ax = pl.subplot(111)

    pl.scatter(X[:,0], X[:,1], c=colors)
    pl.xticks([])
    pl.yticks([])

    if title is not None:
        pl.title(title)


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
rp = random_projection.SparseRandomProjection(n_components=2, random_state=0)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the Iris Data")

#----------------------------------------------------------------------
# Projection on to the first 2 principal components
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca, "Principal Components Projection of the Iris Data")

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
X_lda = lda.LDA(n_components=2).fit_transform(X2, Y)
plot_embedding(X_lda, "Linear Discriminant Projection of the Iris Data")

#----------------------------------------------------------------------
# Isomap projection of the digits dataset
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
plot_embedding(X_iso, "Isomap Projection of the Iris Data")


#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle, "Locally Linear Embedding of the Iris Data")


#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle, "Modified Locally Linear Embedding of the Iris Data")


#----------------------------------------------------------------------
# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,method='hessian')
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle, "Hessian Locally Linear Embedding of the Iris Data")


#----------------------------------------------------------------------
# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa, "Local Tangent Space Alignment of the Iris Data")

#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds, "MDS embedding of the Iris Data")

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding(X_reduced, "Random forest embedding of the Iris Data")


#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
X_se = embedder.fit_transform(X)
plot_embedding(X_se, "Spectral embedding of the Iris Data")


pl.show()

#----------------------------------------------------------------------
# Projection on to the first 3 principal components
fig = pl.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(dataset.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=colors)
ax.set_title("First Three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

pl.show()


"""
Code borrowed in part from plot_compare_methods.py
Author Jake Vanderplas
http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#example-manifold-plot-compare-methods-py
"""

fig = pl.figure(figsize=(15, 8))
pl.suptitle("Manifold Learning with %i points, %i neighbors"
            % (n_samples, n_neighbors), fontsize=14)

try:
    # compatibility matplotlib < 1.0
    ax = fig.add_subplot(241, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)
    ax.view_init(4, -72)
except:
    ax = fig.add_subplot(241, projection='3d')
    pl.scatter(X[:, 0], X[:, 2], c=colors)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
    print("%s" %methods[i])
    ax = fig.add_subplot(242 + i)
    pl.scatter(Y[:, 0], Y[:, 1], c=colors)
    pl.title("%s" %labels[i])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    pl.axis('tight')

Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
ax = fig.add_subplot(246)
pl.scatter(Y[:, 0], Y[:, 1], c=colors)
pl.title("Isomap")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')

mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
ax = fig.add_subplot(247)
pl.scatter(Y[:, 0], Y[:, 1], c=colors)
pl.title("MDS")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')

se = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
Y = se.fit_transform(X)
ax = fig.add_subplot(248)
pl.scatter(Y[:, 0], Y[:, 1], c=colors)
pl.title("SpectralEmbedding")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
pl.axis('tight')

pl.show()
