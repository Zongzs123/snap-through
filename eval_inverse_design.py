import tensorflow as tf
import numpy as np
from utils import load_data
from modules import forward_network, inverse_network

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras import backend as K
import pandas as pd


def main(args):
    train_data = load_data(args.train_path)
    test_data = load_data(args.test_path)
    tmp_data_x = train_data[:, :4]
    tmp_data_y = train_data[:, 6:]
    data_x = test_data[:, :4]
    data_y = test_data[:, 6:]

    # transform the test data into the format of training data
    scaler1 = StandardScaler()
    scaler1.fit(tmp_data_x)
    scaler2 = StandardScaler()
    scaler2.fit(tmp_data_y)
    data_x_standarized = scaler1.transform(data_x)
    data_y_standarized = scaler2.transform(data_y)

    # one-hot encoding
    tmp_array = []
    for i in range(data_x.shape[0]):
        tmp = data_x_standarized[i, :].tolist()
        tmp += to_categorical(test_data[i, 4] - 1, 4).tolist()
        tmp += to_categorical(test_data[i, 5] - 1, 4).tolist()
        tmp_array += [tmp]
    data_x_one_hot = np.array(tmp_array, dtype=np.float32)
    data_standarized = np.append(data_x_one_hot, data_y_standarized, axis=-1)
    print(data_standarized.shape)

    test_data_x = data_standarized[:, :12]
    test_data_y = data_standarized[:, 12:]

    inverse_model_1 = inverse_network()
    inverse_model_2 = inverse_network()
    inverse_model_3 = inverse_network()
    inverse_model_4 = inverse_network()
    inverse_model_5 = inverse_network()
    inverse_model_6 = inverse_network()
    forward_model = forward_network()

    # specify the inverse network 1 and obtain the result of inverse network 1
    inverse_model_1.load_weights(args.inverse_model_loc_1, by_name=False)
    inverse_model_1.trainable = False
    predicted_x_before_transform_1 = inverse_model_1(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_1[:, 4:])
    for i in range(predicted_x_before_transform_1.shape[0]):
        j = np.argmax(predicted_x_before_transform_1[i, 4:8])
        k = np.argmax(predicted_x_before_transform_1[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_1 = np.concatenate((predicted_x_before_transform_1[:, :4], softmax_result), axis=-1)

    # specify the inverse network 2 and obtain the result of inverse network 2
    inverse_model_2.load_weights(args.inverse_model_loc_2, by_name=False)
    inverse_model_2.trainable = False
    predicted_x_before_transform_2 = inverse_model_2(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_2[:, 4:])
    for i in range(predicted_x_before_transform_2.shape[0]):
        j = np.argmax(predicted_x_before_transform_2[i, 4:8])
        k = np.argmax(predicted_x_before_transform_2[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_2 = np.concatenate((predicted_x_before_transform_2[:, :4], softmax_result), axis=-1)

    # specify the inverse network 3 and obtain the result of inverse network 3
    inverse_model_3.load_weights(args.inverse_model_loc_3, by_name=False)
    inverse_model_3.trainable = False
    predicted_x_before_transform_3 = inverse_model_3(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_3[:, 4:])
    for i in range(predicted_x_before_transform_3.shape[0]):
        j = np.argmax(predicted_x_before_transform_3[i, 4:8])
        k = np.argmax(predicted_x_before_transform_3[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_3 = np.concatenate((predicted_x_before_transform_3[:, :4], softmax_result), axis=-1)

    # specify the inverse network 4 and obtain the result of inverse network 4
    inverse_model_4.load_weights(args.inverse_model_loc_4, by_name=False)
    inverse_model_4.trainable = False
    predicted_x_before_transform_4 = inverse_model_4(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_4[:, 4:])
    for i in range(predicted_x_before_transform_4.shape[0]):
        j = np.argmax(predicted_x_before_transform_4[i, 4:8])
        k = np.argmax(predicted_x_before_transform_4[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_4 = np.concatenate((predicted_x_before_transform_4[:, :4], softmax_result), axis=-1)

    # specify the inverse network 5 and obtain the result of inverse network 5
    inverse_model_5.load_weights(args.inverse_model_loc_5, by_name=False)
    inverse_model_5.trainable = False
    predicted_x_before_transform_5 = inverse_model_5(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_5[:, 4:])
    for i in range(predicted_x_before_transform_5.shape[0]):
        j = np.argmax(predicted_x_before_transform_5[i, 4:8])
        k = np.argmax(predicted_x_before_transform_5[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_5 = np.concatenate((predicted_x_before_transform_5[:, :4], softmax_result), axis=-1)

    # specify the inverse network 6 and obtain the result of inverse network 6
    inverse_model_6.load_weights(args.inverse_model_loc_6, by_name=False)
    inverse_model_6.trainable = False
    predicted_x_before_transform_6 = inverse_model_6(test_data_y, training=False)
    softmax_result_temp = np.array(predicted_x_before_transform_6[:, 4:])
    for i in range(predicted_x_before_transform_6.shape[0]):
        j = np.argmax(predicted_x_before_transform_6[i, 4:8])
        k = np.argmax(predicted_x_before_transform_6[i, 8:12])
        softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
        softmax_result_temp[i, j] = 1
        softmax_result_temp[i, k + 4] = 1
    softmax_result = np.array(softmax_result_temp, dtype=np.float32)
    predicted_x_before_transform_6 = np.concatenate((predicted_x_before_transform_6[:, :4], softmax_result), axis=-1)

    # specify the forward model and obtain the result of forward model
    forward_model.load_weights(args.forward_model_loc, by_name=False)
    forward_model.trainable = False
    predicted_y_before_transform_1 = forward_model(predicted_x_before_transform_1, training=False)
    predicted_y_before_transform_2 = forward_model(predicted_x_before_transform_2, training=False)
    predicted_y_before_transform_3 = forward_model(predicted_x_before_transform_3, training=False)
    predicted_y_before_transform_4 = forward_model(predicted_x_before_transform_4, training=False)
    predicted_y_before_transform_5 = forward_model(predicted_x_before_transform_5, training=False)
    predicted_y_before_transform_6 = forward_model(predicted_x_before_transform_6, training=False)

    # obtain the predicted data using inverse network 1
    predicted_x_points_1 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_1[:, :4]), predicted_x_before_transform_1[:, 4:]),
        axis=-1)
    predicted_y_points_1 = scaler2.inverse_transform(predicted_y_before_transform_1)

    # obtain the predicted data using inverse network 2
    predicted_x_points_2 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_2[:, :4]), predicted_x_before_transform_2[:, 4:]),
        axis=-1)
    predicted_y_points_2 = scaler2.inverse_transform(predicted_y_before_transform_2)

    # obtain the predicted data using inverse network 3
    predicted_x_points_3 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_3[:, :4]), predicted_x_before_transform_3[:, 4:]),
        axis=-1)
    predicted_y_points_3 = scaler2.inverse_transform(predicted_y_before_transform_3)

    # obtain the predicted data using inverse network 4
    predicted_x_points_4 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_4[:, :4]), predicted_x_before_transform_4[:, 4:]),
        axis=-1)
    predicted_y_points_4 = scaler2.inverse_transform(predicted_y_before_transform_4)

    # obtain the predicted data using inverse network 5
    predicted_x_points_5 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_5[:, :4]), predicted_x_before_transform_5[:, 4:]),
        axis=-1)
    predicted_y_points_5 = scaler2.inverse_transform(predicted_y_before_transform_5)

    # obtain the predicted data using inverse network 6
    predicted_x_points_6 = np.concatenate(
        (scaler1.inverse_transform(predicted_x_before_transform_6[:, :4]), predicted_x_before_transform_6[:, 4:]),
        axis=-1)
    predicted_y_points_6 = scaler2.inverse_transform(predicted_y_before_transform_6)

    # select the best prediction and evaluate using our defined metrics
    error_ratio_array = []
    area_ground_truth = []
    predicted_x_points = []
    predicted_y_points = []
    for i in range(data_y.shape[0]):
        inverse_design_predicted_error_1 = K.sum(K.abs(predicted_y_points_1[i, :] - data_y[i, :]))
        inverse_design_predicted_error_2 = K.sum(K.abs(predicted_y_points_2[i, :] - data_y[i, :]))
        inverse_design_predicted_error_3 = K.sum(K.abs(predicted_y_points_3[i, :] - data_y[i, :]))
        inverse_design_predicted_error_4 = K.sum(K.abs(predicted_y_points_4[i, :] - data_y[i, :]))
        inverse_design_predicted_error_5 = K.sum(K.abs(predicted_y_points_5[i, :] - data_y[i, :]))
        inverse_design_predicted_error_6 = K.sum(K.abs(predicted_y_points_6[i, :] - data_y[i, :]))
        error_ratio_1 = tf.cast(inverse_design_predicted_error_1, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_2 = tf.cast(inverse_design_predicted_error_2, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_3 = tf.cast(inverse_design_predicted_error_3, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_4 = tf.cast(inverse_design_predicted_error_4, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_5 = tf.cast(inverse_design_predicted_error_5, dtype=tf.float32) / (K.sum(data_y[i, :]))
        error_ratio_6 = tf.cast(inverse_design_predicted_error_6, dtype=tf.float32) / (K.sum(data_y[i, :]))
        best_flag = np.argmin(
            [error_ratio_1, error_ratio_2, error_ratio_3, error_ratio_4, error_ratio_5, error_ratio_6])
        if best_flag == 0:
            error_ratio_array.append(error_ratio_1)
            predicted_x_points.append(predicted_x_points_1[i, :])
            predicted_y_points.append(predicted_y_points_1[i, :])
        elif best_flag == 1:
            error_ratio_array.append(error_ratio_2)
            predicted_x_points.append(predicted_x_points_2[i, :])
            predicted_y_points.append(predicted_y_points_2[i, :])
        elif best_flag == 2:
            error_ratio_array.append(error_ratio_3)
            predicted_x_points.append(predicted_x_points_3[i, :])
            predicted_y_points.append(predicted_y_points_3[i, :])
        elif best_flag == 3:
            error_ratio_array.append(error_ratio_4)
            predicted_x_points.append(predicted_x_points_4[i, :])
            predicted_y_points.append(predicted_y_points_4[i, :])
        elif best_flag == 4:
            error_ratio_array.append(error_ratio_5)
            predicted_x_points.append(predicted_x_points_5[i, :])
            predicted_y_points.append(predicted_y_points_5[i, :])
        else:
            error_ratio_array.append(error_ratio_6)
            predicted_x_points.append(predicted_x_points_6[i, :])
            predicted_y_points.append(predicted_y_points_6[i, :])

        area_ground_truth.append(K.sum(data_y[i, :]))

    error_ratio_df = pd.DataFrame(np.array(error_ratio_array))
    area_ground_truth_df = pd.DataFrame(np.array(area_ground_truth))
    error_ratio_df.to_csv('test_error_ratio.csv', index=False)  # the name can be changed for different purposes
    area_ground_truth_df.to_csv('test_area.csv', index=False)  # the name can be changed for different purposes
    mean_error_ratio = tf.reduce_mean(error_ratio_array)
    print(len(error_ratio_array))
    print(mean_error_ratio)

    predicted_x_points = np.array(predicted_x_points)
    predicted_y_points = np.array(predicted_y_points)
    predicted_points = np.concatenate((predicted_x_points, predicted_y_points), axis=-1)
    print(predicted_points.shape)
    predicted_points_df = pd.DataFrame(predicted_points)
    predicted_points_df.to_csv('predicted_test.csv', index=False)  # the name can be changed for different purposes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_path = "train.xlsx"
    args.test_path = "test.xlsx"
    args.inverse_model_loc_1 = "inverse_model_0.keras"
    args.inverse_model_loc_2 = "inverse_model_1.keras"
    args.inverse_model_loc_3 = "inverse_model_3.keras"
    args.inverse_model_loc_4 = "inverse_model_4.keras"
    args.inverse_model_loc_5 = "inverse_model_6.keras"
    args.inverse_model_loc_6 = "inverse_model_8.keras"
    args.forward_model_loc = "forward_model.keras"
    main(args)
