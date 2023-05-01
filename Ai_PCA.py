import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd


# Create random 2d data
mu = np.array([10,13])
sigma = np.array([[3.5, -1.8], [-1.8,3.5]])

print("Mu ", mu.shape)
print("Sigma ", sigma.shape)

# Create 1000 samples using mean and sigma
org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
print("Data shape ", org_data.shape)
print(org_data)

plt.subplot(1, 4, 1)
plt.title('origin data')
for x, y in org_data:
    plt.scatter(x, y, c="red", s=1)


# Subtract mean from data
mean = np.mean(org_data, axis = 0)
print("Mean ", mean.shape)
mean_data = org_data - mean
print("Data after subtracting mean ", mean_data.shape, "\n")

plt.subplot(1, 4, 2)
plt.title('mean data')
for x, y in mean_data:
    plt.scatter(x, y, c="red", s=1)


# Compute covariance matrix 공분산
cov = np.cov(mean_data.T)
cov = np.round(cov, 2)# 소수점 2번째까지 줄임
print("Covariance matrix ", cov.shape, "\n")

# Perform eigen decomposition of covariance matrix
eig_val, eig_vec = np.linalg.eig(cov)
print("Eigen vectors ", eig_vec)
print("Eigen values ", eig_val, "\n")

# Sort eigen values and corresponding eigen vectors in descending order 내림 정렬
indices = np.arange(0, len(eig_val), 1)
indices = ([x for _, x in sorted(zip(eig_val, indices))])[::-1]
eig_val = eig_val[indices]
eig_vec = eig_vec[:, indices]
print("Sorted Eigen vectors ", eig_vec)
print("Sorted Eigen values ", eig_val, "\n")

# Get explained variance 축 선택
sum_eig_val = np.sum(eig_val)
explained_variance = eig_val/ sum_eig_val
print(explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print(cumulative_variance) # 70% 넘는 값이 옳게된 모델

# Take transpose of eigen vectors with data
pca_data = np.dot(mean_data, eig_vec)
print("Transformed data ", pca_data.shape)

plt.subplot(1, 4, 3)
plt.title('tranform data')
for x, y in pca_data:
    plt.scatter(x, y, c="red", s=1)

pca_data_mean = pca_data + mean
print("pca_data_mean data ", pca_data_mean.shape)

plt.subplot(1, 4, 4)
plt.title('final data')
for x, y in pca_data_mean:
    plt.scatter(x, y, c="red", s=1)

plt.show()
