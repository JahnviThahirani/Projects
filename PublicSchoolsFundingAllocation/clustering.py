import csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_csv(file_in):
    with open(file_in, 'r', newline='') as fin:
        csvin = csv.reader(fin)
        lines = [line[1:] for line in csvin]
    return lines

def read_PCA(file_in):
    with open(file_in, 'r', newline='') as fin:
        csvin = csv.reader(fin)
        lines = [line[-2:] for line in csvin]
    return lines

def append_features(col):
    with open('grouped_data_2016_pca.csv', 'w+') as fo:
        csvwriter = csv.writer(fo, delimiter=',')
        csvwriter.writerows(col)

def append_clusters(file, col):
    with open(file, 'r', newline='') as fi:
        csvin = csv.reader(fi)
        lines = [line for line in csvin]
    new_lines = [line + [str(col[i])] for i, line in enumerate(lines)]

    with open('grouped_data_2016_kmeans.csv', 'w') as fo:
        for line in new_lines:
            fo.write(','.join(line) + '\n')

# data = read_csv("grouped_data_2016.csv")[1:]
# standard_data = StandardScaler().fit(data).transform(data)
# pca_data = PCA(n_components=2).fit_transform(standard_data).tolist()
# pca_data.insert(0, ['feature1', 'feature2'])
# append_features(pca_data)

updated_data = read_PCA('grouped_data_2016.csv')[1:]
kmeans = KMeans(n_clusters=4, random_state=0).fit(updated_data)

clusters = kmeans.labels_.tolist()
clusters.insert(0, 'clusters')
append_clusters('grouped_data_2016.csv', clusters)


