import tensorflow as tf
from keras.callbacks import *
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from utils import load_data, inverse_augmentation
from modules import inverse_network

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
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
    data_augmented = inverse_augmentation(data_standarized)
    print(data_augmented.shape)
    # inputs represents the curve features, targets represents the structural parameters
    np.random.shuffle(data_augmented)
    targets = data_augmented[:, :12]
    inputs = data_augmented[:, 12:]

    # Define the hyperparameter search space
    # Note that randomized search and grid search should use different dict !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    param_dist = {
        'units_layer1': [64, 100, 128],
        'units_layer2': [400, 512, 600],
        'optimizer': ['adam', 'rmsprop'],
    }

    # Create the inverse_model
    inverse_model = KerasRegressor(build_fn=inverse_network, epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    # Create a scorer for the RandomizedSearchCV based on custom loss_MAE
    scorer = make_scorer(loss_mae, greater_is_better=False)

    # Define a learning rate scheduler callback
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

    if args.flag == 1:
        # Create the RandomizedSearchCV object with the callbacks
        random_search = RandomizedSearchCV(inverse_model, param_distributions=param_dist, n_iter=25, cv=5, scoring=scorer,
                                           verbose=2)
    else:
        # Create the GridSearchCV object with the callbacks
        random_search = GridSearchCV(inverse_model, param_dist, cv=5, scoring=scorer, verbose=2)

    print('------------------------------------------------------------------------')
    print(f'Training ...')

    # Fit the RandomizedSearchCV object to the data
    random_search.fit(inputs, targets, callbacks=[lr_scheduler])

    # Print the best hyperparameters
    print("Best Hyperparameters: ", random_search.best_params_)
    best_hyperparameters = random_search.best_params_
    print("all the result of CV: ", random_search.cv_results_)

    # Create a new model with the best hyperparameters
    best_inverse_model = inverse_network(units_layer1=best_hyperparameters['units_layer1'],
                                         units_layer2=best_hyperparameters['units_layer2'],
                                         optimizer=best_hyperparameters['optimizer'])

    # Train again the model to obtain the history
    train_result = best_inverse_model.fit(inputs, targets, epochs=args.epochs, batch_size=args.batch_size,
                                          callbacks=[lr_scheduler], verbose=2, validation_split=0.2)

    # save the model of best hyperparameters
    base_dir = args.run_name
    os.makedirs(base_dir, exist_ok=True)
    model_loc = os.path.join(base_dir, 'inverse_model.keras')
    best_inverse_model.save(model_loc)

    # convert cv_results_ to dataframe
    cv_results_df = pd.DataFrame(random_search.cv_results_)

    # Save cv_results_ to a CSV file
    cv_results_file = os.path.join(base_dir, 'cv_results.csv')
    cv_results_df.to_csv(cv_results_file, index=False)

    # Print all the results of CV
    print("All the results of CV:\n", cv_results_df)

    # plot the figure of training result
    loss_MAE = train_result.history['loss_mae']
    val_loss_MAE = train_result.history['val_loss_mae']
    loss = train_result.history['loss']
    val_loss = train_result.history['val_loss']
    epochs = range(len(loss_MAE))

    plt.plot(epochs, loss_MAE, 'b', label='Training absolute mean error')
    plt.plot(epochs, val_loss_MAE, 'r', label='Validation absolute mean error')
    plt.title('Training and validation absolute mean error')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "./inverse_network_grid/20240514/"
    args.epochs = 300
    args.batch_size = 64
    args.dataset_path = "train.xlsx"
    args.flag = 0  # 1 for random search, 0 for grid search
    train(args)
