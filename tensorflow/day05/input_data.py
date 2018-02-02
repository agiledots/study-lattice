import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data_filename = "./data/data_train.csv"
eval_data_filename = "./data/data_eval.csv"
predict_data_filename = "./data/data_predict.csv"

def make_csv(filename):
    # x = np.random.uniform(-5, 5, 50)
    x = np.linspace(-5, 5, 50)
    y = np.square(x) + np.random.normal(0, 2.5, x.shape)

    data = np.array([0.0, 0.0])
    for (a, b) in zip(x, y):
        data = np.vstack((data, [a, b]))

    data = np.delete(data, 0, 0)

    # データフレームを作成
    df = pd.DataFrame(
        data,
        columns=['x', 'y'],
    )
    print(data)

    # CSV ファイル として出力
    df.to_csv(filename,
              encoding='utf-8',
              index=False,
              header=True
              )

# 
def create_data():
    make_csv(train_data_filename)
    make_csv(eval_data_filename)
    make_csv(predict_data_filename)


def read_data(type):
    filename = None

    if type == 'train':
        filename = train_data_filename
    elif type == "eval":
        filename = eval_data_filename
    elif type == 'predict':
        filename = predict_data_filename

    if filename is not None :
        data = pd.read_csv(filename, dtype={'x': np.float32, 'y': np.float32})
        return data.values


def show(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    #draw function line
    x = np.arange(-5.1, 5.1, 0.1)
    y = np.square(x)
    ax1.plot(x, y, "r-", lw=5, label="fucntion(y=x^2)")

    # draw dots
    ax1.plot(data.T[0], data.T[1], 'o', label='Original data')
    plt.legend()

    # show
    plt.show()

if __name__ == '__main__':
    create_data()
    data = read_data("train")
    show(data)