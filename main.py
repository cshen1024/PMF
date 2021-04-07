import numpy as np

from utils import data_process, RMSE
from model import PMF


if __name__=='__main__':
    # params
    lambda_alpha = 0.01
    lambda_beta = 0.01
    latent_size = 20
    lr = 3e-5
    iters = 100

    # load data
    data_path = './data/ml-100k/ratings.dat'
    dict_userid_to_index, dict_itemid_to_index, data = data_process(data_path)

    # split train, valid and test data
    ratio = 0.7
    train_data = data[:int(ratio*data.shape[0])]
    vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]
    test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]

    # complete rating matrix
    rows = max(dict_userid_to_index.values()) + 1
    columns = max(dict_itemid_to_index.values()) + 1

    R = np.zeros((rows, columns))
    for tuple in train_data:
        R[int(tuple[0]), int(tuple[1])] = float(tuple[2])

    # model
    model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta,
                latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)

    U, V, train_loss_list, vali_rmse_list = model.train(train_data=train_data, vali_data=vali_data)

    preds = model.predict(data=test_data)
    test_rmse = RMSE(preds, test_data[:, 2])

    print('test rmse:{}'.format(test_rmse))















