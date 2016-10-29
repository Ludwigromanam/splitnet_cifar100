#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sklearn.cluster as clu
import cPickle as pickle


ckpt_fpath = './model.ckpt-80000'
fc_weight_name = 'logits/fc/weights'
cifar100_class_fname = 'fine_label_names.txt'
output_fname = 'clustering.pkl'

num_cluster1 = 2
num_cluster2 = 5

# Spectral clustering
def spectral_clustering(X, n_clusters):
    spectral_clu = clu.SpectralClustering(n_clusters=n_clusters)
    y = spectral_clu.fit_predict(X)
    clusters = [[] for _ in range(n_clusters)]
    for i in range(X.shape[0]):
        clusters[y[i]].append(i)

    cluster_centers = []
    for i in range(n_clusters):
        cluster_centers.append(np.average(X[clusters[i],:], axis=0))
    cluster_centers = np.array(cluster_centers)

    return y, clusters, cluster_centers

# Load CIFAR-100 class names
print('Load CIFAR-100 class names')
with open(cifar100_class_fname) as fd:
    classes = [temp.strip() for temp in fd.readlines()]

# Open TensorFlow ckpt and load the last weight
print('Load tensor: %s' % fc_weight_name)
reader = tf.train.NewCheckpointReader(ckpt_fpath)
weight = reader.get_tensor(fc_weight_name)
weight = weight.transpose()

# Clustering
print('Clustering...\n')
y2, clusters2, centers2 = spectral_clustering(weight, num_cluster2)
y1, clusters1, centers1 = spectral_clustering(centers2, num_cluster1)

output = []
for i1, c1 in enumerate(clusters1):
    temp1 = []
    for i2, ci2 in enumerate(c1):
        print 'Cluster %d-%d: ' % (i1+1, i2+1) ,
        for idx in clusters2[ci2]:
            print classes[idx] ,
        print ' '
        temp1.append(clusters2[ci2])
    output.append(temp1)
print ' '

# Save as pkl file
print('Save as pkl file')
with open(output_fname, 'wb') as fd:
    pickle.dump(output, fd)

print('Done!')
