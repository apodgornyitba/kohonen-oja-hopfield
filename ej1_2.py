from oja import OjaPerceptron
from sklearn.preprocessing import StandardScaler
from config import load_oja_config
from matplotlib import pyplot as plt
import numpy as np


from utils.parser import parse_csv_file

df = parse_csv_file('./inputs/europe.csv')
countries = df["Country"].to_numpy()
df.drop(columns=["Country"], axis=1, inplace=True)
cells = list(df.columns)
inputs = StandardScaler().fit_transform(df.values)

config = load_oja_config()

epochs = int(config['epochs'])
learning_rate = float(config['learning_rate'])

oja = OjaPerceptron(inputs, learning_rate)
pca1 = oja.train(epochs)
pca1 = np.array(pca1)
pca1 = np.multiply(pca1, -1)

print(f"Oja eigenvector that builds PC1:\n {pca1}")

libray_pca1 = [0.12487390183337656,-0.5005058583604993,0.4065181548118897,-0.4828733253002008,0.18811161613179747,-0.475703553912758,0.27165582007504635]


# print(f"Oja eigenvector that builds PC1:\n {pca1}")
# countries_pca1 = [np.inner(pca1,inputs[i]) for i in range(len(inputs))]
# countries_library_pca1 = [-np.inner(libray_pca1,inputs[i]) for i in range(len(inputs))]
# fig,(ax1,ax2) = plt.subplots(1,2, figsize=(12, 10))
# bar1 = ax1.bar(countries,countries_pca1)
# bar2 = ax2.bar(countries,countries_library_pca1)
# ax1.set_ylabel('PCA1')
# ax1.set_title('PCA1 per country using Oja')
# ax2.set_ylabel('PCA1')
# ax2.set_title('PCA1 per country using Sklearn')
# ax1.set_xticks(range(len(countries)))
# ax2.set_xticks(range(len(countries)))
# ax1.set_xticklabels(countries, rotation=90)
# ax2.set_xticklabels(countries, rotation=90)
# plt.show()

# plot errors of oja and library changing learning rate
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

oja_errors = []
for lr in learning_rates:
    acum_error = 0
    for i in range(5):
        oja = OjaPerceptron(inputs, lr)
        pca1 = oja.train(epochs)
        pca1 = np.array(pca1)
        pca1 = np.multiply(pca1, -1)
        sum_error = abs(pca1 - libray_pca1).sum()
        error = sum_error / len(pca1)
        acum_error += error
    oja_errors.append(acum_error / 5)



# plot errors of oja and library
fig,ax = plt.subplots(1,1, figsize=(12, 10))
x_pos = np.arange(len(learning_rates))
bar = ax.bar(x_pos,oja_errors)
ax.set_ylabel('Error')
ax.set_xlabel('Learning rate')
ax.set_title('Error of Oja vs Learning rate')
ax.set_xticks(x_pos)
ax.set_xticklabels(learning_rates)
plt.show()