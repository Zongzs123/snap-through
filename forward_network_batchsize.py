import math
import tensorflow as tf
from keras.callbacks import *
import numpy as np
from utils import load_data, forward_augmentation
from modules import forward_network

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import pandas as pd


def train(args):
    from utils import loss_mae
    data = load_data(args.dataset_path)  # data_x represents structural parameters and data_y represents curve features
    data_x = data[:, :4]
    data_y = data[:, 6:]
    scaler1 = StandardScaler()
    scaler1.fit(data_x)
    data_x_standarized = scaler1.transform(data_x)

    scaler2 = StandardScaler()
    scaler2.fit(data_y)
    data_y_standarized = scaler2.transform(data_y)

    tmp_array = []
    for i in range(data_x.shape[0]):
        tmp = data_x_standarized[i, :].tolist()
        tmp += to_categorical(data[i, 4] - 1, 4).tolist()
        tmp += to_categorical(data[i, 5] - 1, 4).tolist()
        tmp_array += [tmp]
    data_x_one_hot = np.array(tmp_array, dtype=np.float32)
    data_standarized = np.append(data_x_one_hot, data_y_standarized, axis=-1)
    data_augmented = forward_augmentation(data_standarized)
    print(data_augmented.shape)
    # inputs represents the structural parameters, targets represents the curve features
    np.random.shuffle(data_augmented)
    inputs = data_augmented[:, :12]
    targets = data_augmented[:, 12:]

    # Define a learning rate scheduler callback
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

    print('------------------------------------------------------------------------')
    print(f'Training ...')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    loss = []
    val_loss = []
    loss_MAE = []
    val_loss_MAE = []

    # train the model using kfold cross-validation
    for train_index, val_index in kfold.split(inputs):
        train_inputs, train_targets = inputs[train_index, :], targets[train_index, :]
        val_inputs, val_targets = inputs[val_index, :], targets[val_index, :]

        forward_model = forward_network()
        forward_model.summary()

        # Train again the model to obtain the history
        train_result = forward_model.fit(train_inputs, train_targets, epochs=args.epochs, batch_size=args.batch_size,
                                         callbacks=[lr_scheduler], verbose=2, validation_data=(val_inputs, val_targets),
                                         shuffle=True)

        loss.append(train_result.history['loss'])
        val_loss.append(train_result.history['val_loss'])
        loss_MAE.append(train_result.history['loss_mae'])
        val_loss_MAE.append(train_result.history['val_loss_mae'])

    loss = np.array(loss)
    val_loss = np.array(val_loss)
    loss_MAE = np.array(loss_MAE)
    val_loss_MAE = np.array(val_loss_MAE)

    loss_df = pd.DataFrame(loss.T)
    val_loss_df = pd.DataFrame(val_loss.T)
    loss_MAE_df = pd.DataFrame(loss_MAE.T)
    val_loss_MAE_df = pd.DataFrame(val_loss_MAE.T)

    # save the model of best hyperparameters
    base_dir = args.run_name
    os.makedirs(base_dir, exist_ok=True)

    # Save cv_results_ to a CSV file
    cv_results_file = os.path.join(base_dir, 'loss.csv')
    loss_df.to_csv(cv_results_file, index=False)
    cv_results_file = os.path.join(base_dir, 'val_loss.csv')
    val_loss_df.to_csv(cv_results_file, index=False)
    cv_results_file = os.path.join(base_dir, 'loss_MAE.csv')
    loss_MAE_df.to_csv(cv_results_file, index=False)
    cv_results_file = os.path.join(base_dir, 'val_loss_MAE.csv')
    val_loss_MAE_df.to_csv(cv_results_file, index=False)

    # printing the loss_MAE and val_loss_MAE results
    logging_info = "loss_batch_size" + str(args.batch_size) + "_mean:\n"
    print(logging_info, np.mean(loss[:, args.epochs - 1]))
    logging_info = "val_loss_batch_size" + str(args.batch_size) + "_mean:\n"
    print(logging_info, np.mean(val_loss[:, args.epochs - 1]))
    logging_info = "loss_MAE_batch_size" + str(args.batch_size) + "_mean:\n"
    print(logging_info, np.mean(loss_MAE[:, args.epochs - 1]))
    logging_info = "val_loss_MAE_batch_size" + str(args.batch_size) + "_mean:\n"
    print(logging_info, np.mean(val_loss_MAE[:, args.epochs - 1]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = "train.xlsx"
    args.epochs = 200
    for i in range(5):
        args.batch_size = int(16 * math.pow(2, i))
        args.run_name = "./forward_network_batch_size" + str(args.batch_size) + "/"
        train(args)
