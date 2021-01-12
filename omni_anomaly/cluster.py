import os
import sys
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
"""
	python3
	cluster the machine
"""

INFLECT_THRESH = 0.1 #0.01, 0.1, 1
graphDir = "../graph/"


def inflection_point(radius, k_dis, start, end, threshold):
	"""
	k_dis is a sorted list, the curve is monotonically increasing. The max Y-value is smaller than two, so the diff
	between two parts of curve is also smaller than 2.
	"""
	r = -1
	diff = sys.maxsize
	if end - start <= 2:#0, 2 may bring a bug
		return
	for i in range(start, end):
		if i == start:
			left = 0
		else:
			left = (k_dis[i] - k_dis[start]) / (i - start)
		right = (k_dis[end] - k_dis[i]) / (end - i)
		if left > 0.01 or right > 0.01:
			continue
		if abs(right - left) < diff:
			diff = abs(right - left)
			r = i
	if diff < threshold:
		radius.append(r)
	inflection_point(radius, k_dis, start, r-1, threshold)
	inflection_point(radius, k_dis, r+1, end, threshold)


def density_estimation(sim_matrix, inflect_thresh, min_samples, EPS_para): #inflect_thresh:(2k:0.001)
	"""
	calculate (min_samples - 1)-NN similarkity distance of the training data and determine the density radius.
	"""
	k_dis = []
	for id in np.arange(0, len(sim_matrix[0])):
		sim_list = sim_matrix[id][:]
		sorted_sim = np.sort(sim_list)
		k_dis.append(sorted_sim[min_samples-1])
	k_dis.sort(reverse=True)

	sorted_k_dis = k_dis/np.max(k_dis) #K-dis curve
	length = len(sorted_k_dis)

	radius = [] # all candidate density radius list
	inflection_point(radius, sorted_k_dis, 0, length-1, threshold=inflect_thresh)

	ra_vals = []
	for ra in radius:
		ra_vals.append(sorted_k_dis[ra]*np.max(k_dis))
	print(ra_vals)

	if len(ra_vals) > 0:
		t_ra_vals = [i for i in ra_vals if i < EPS_para]
		if len(t_ra_vals) > 0:
			return max(t_ra_vals)
		else:
			return EPS_para
	else:
		return EPS_para


def myDBSCAN(distMatirx, MINSAMPLE, EPS_PARA):
	"""
	dbscan using the precomputed distance matrix
	"""
	epsV = density_estimation(distMatirx, INFLECT_THRESH, MINSAMPLE, EPS_PARA) # 4
	print('Density cluster radius: %f' % epsV)
	print(distMatirx, MINSAMPLE, EPS_PARA)

	db = DBSCAN(eps=epsV, min_samples=MINSAMPLE, metric="precomputed")
	db.fit(distMatirx)
	labels = db.labels_
	no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	print(labels, no_clusters)
	print(np.sum(labels > 0))
	return labels


def aveSilhouetteCoefficient(clusters, distMatirx):
	"""
	calculate the average silhouette coefficient of cluster results
	"""
	scList = []
	# print('No of clusters:', len(clusters))
	# print(clusters)
	for eachC in clusters:
		if len(eachC) == 1:
			scList.append(0)
		else:
			for eachItem in eachC:
				ai = sum([distMatirx[eachItem][otherItem] for otherItem in eachC])/float(len(eachC)-1)
				biList = []
				otherC = [x for x in clusters if x != eachC]
				# print(eachItem, eachC, otherC)
				for eachCInOtherC in otherC:
					biList.append(sum([distMatirx[eachItem][otherItemInC] for otherItemInC in eachCInOtherC])/float(len(eachCInOtherC)))
				bi = min(biList)
				si = (bi - ai)/float(max(bi, ai))
				scList.append(si)
	return np.mean(scList)


def getClusterResultOfAGG(labelArray, clusterNumber):
	print("-"*30)
	print("No of clusters:%d"%clusterNumber)
	clustersList = []
	for i in range(clusterNumber):
		clusterIndices = list(np.nonzero(labelArray == i)[0])
		clustersList.append(clusterIndices)
		print('Cluster %d: %s'%(i, clusterIndices))
	return(clustersList)
