from line_profiler import LineProfiler
from sklearn.datasets import make_spd_matrix
from mxnet.test_utils import set_default_context
import mxnet as mx
from mxnet import npx
import csv
import time

# define the mean points for each of the systhetic cluster centers
means = [[8.4, 8.2], [1.4, 1.6], [2.4, 5.4], [6.4, 2.4]]

# for each cluster center, create a Positive semidefinite convariance matrix
covs = []
for s in range(len(means)):
  covs.append(make_spd_matrix(2))

theta = []
for mean, cov in zip(means, covs):
    theta.append([mean, cov])

print(theta)

# Generate Training Data
def GMM_Generate(K, N, means, covs, size):
    import numpy as np
    train_data = []
    for i in range(K):
        mean_k = means[i] 
        cov_k = covs[i]
        data = np.random.multivariate_normal(mean_k, cov_k, size=size)
        train_data += list(data)
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
    np.savetxt("GMM_Train_Data_{}.csv".format(K*size), np.atleast_2d(train_data), fmt='%.8f', delimiter=',')   
    return train_data

def test_GMM(op_type, K, train_data, trails):
    print("Start fitting")
    time_start = time.time()
    for i in range(trails):
        gmm1 = GaussianMixtureModel(k=K, tolerance=1e-5)  
        sample_assignments = gmm1.predict(train_data)
    time_end = time.time()
    print("End fitting")
    print("-------------------------------------------")
    print(trails, "trails:", op_type, "in dataset of shape", train_data.shape,
         "consumed: ", time_end - time_start, " seconds")
    if op_type == 'Official Numpy':
        np.savetxt("Official_Numpy_Assignment.csv", sample_assignments.reshape(-1,1), fmt="%i")
        for param in gmm1.parameters:
            print(param)
    else:
        #print(sample_assignments.reshape((-1,1)))
        for param in gmm1.parameters:
            print(param)
    

K = 4  # number of cluster
N = 2  # number of dimensions
size = 500 # number of data for each cluster
#GMM_Generate(K, N, means, covs, size)

op_types = ['Official Numpy', 'DeepNumPy CPU', 'DeepNumPy GPU']
op_type = op_types[2]
trails = 1

if op_type == 'Official Numpy':
    import numpy as np
    from mlfromscratch.unsupervised_learning import GaussianMixtureModel
else:
    from mxnet import numpy as np
    from DeepNumpy_mlfromscratch.unsupervised_learning import GaussianMixtureModel
    if op_type == 'DeepNumPy CPU':
        set_default_context(mx.cpu(0))        
    else: 
        set_default_context(mx.gpu(0))
    npx.set_np()

with open("GMM_Train_Data_{}.csv".format(K*size), newline='') as csvfile:
    train_data = np.array(list((csv.reader(csvfile))),dtype=np.float32)
test_GMM(op_type, K, train_data, trails)
