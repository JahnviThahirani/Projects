import csv
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import math
from sklearn.tree import export_graphviz
from sklearn.linear_model import Lasso


def lm(data, groups):
    models = {}
    for group in groups:
        model = Lasso().fit([row[1:10] for row in data[group][0]], data[group][2])
        models[group] = model
    return models


def rf(data, groups):
    models = {}
    for group in groups:
        x = [row[1:10] for row in data[group][0]]
        model = RandomForestRegressor().fit(x, data[group][2])
        models[group] = model
    return models

def svm(data, groups):
    models = {}
    for group in groups:
        x_train_standardized = StandardScaler().fit_transform(data[group][0])
        models[group] = SVR(gamma='auto').fit(x_train_standardized, data[group][2])
    return models

def read_csv(file_in):

    with open(file_in, 'r', newline='') as fin:
        csvin = csv.reader(fin)
        lines = [line for line in csvin]

    return lines


def write_to_csv(values, labels, file_out):

    with open(file_out, 'w', newline='') as fout:
        csvout = csv.writer(fout)
        csvout.writerow(labels)
        csvout.writerows(values)


data = read_csv("grouped_data_kmeans.csv")[1:]
years = [row[0] for row in data]
groups = sorted(list(set([int(row[-1]) for row in data])))
split_data = []

# Split data into separate groups, testing and training data
for group in groups:
    X_group = [row[1:2]+list(map(float, row[2:7]+row[9:15])) for row in data if group == int(row[-1])]
    # use index = 7 for regular, 8 for ln
    y_group = [float(row[8]) for row in data if group == int(row[-1])]
    # x_train, x_test, y_train, y_test
    splits = train_test_split(X_group, y_group, test_size = .3, random_state = 4242, shuffle = True)
    split_data.append(splits)

linear_model = lm(split_data, groups)
random_forest = rf(split_data, groups)

x = [row[0]+row[1] for row in split_data]
y =[row[2]+row[3] for row in split_data]
lm_predictions = []
rf_predictions = []

for group in groups:
    grid_rf = GridSearchCV(random_forest[group], {'max_depth': [20, 30, 40, 50, 60, 70]}, cv=10).fit([row[1:10] for row in split_data[group][0]], split_data[group][2])
    print(grid_rf.best_params_, grid_rf.best_score_)
    rf_predictions.append(grid_rf.predict([row[1:10] for row in x[group]]))
    prediction = linear_model[group].predict([row[1:10] for row in x[group]])
    lm_predictions.append(prediction)
    # print(r2_score(y[group], prediction))
    # print(linear_model[group].score([row[1:10] for row in split_data[group][0]], split_data[group][2]))


lm_error = sum([[y[j][i] - lm_predictions[j][i] for i in range(len(lm_predictions[j]))] for j in groups], [])
rf_error = sum([[y[j][i] - rf_predictions[j][i] for i in range(len(rf_predictions[j]))] for j in groups], [])
lm_se = [math.pow(lm_error[i], 2) for i in range(len(lm_error))]
rf_se = [math.pow(rf_error[i], 2) for i in range(len(rf_error))]

combined_x = sum(x,[])
output = [[years[i], combined_x[i][0], combined_x[i][-1], combined_x[i][-2], lm_error[i], lm_se[i], rf_error[i], rf_se[i]] for i in range(len(lm_error))]
labels = ["Year", "State", "Group", "Grad_Rate", "LM_Error", "LM_SE", "RF_Error", "RF_SE"]
write_to_csv(output, labels, "all_results.csv")




