# snap-through
machine learning method to inverse design a snap-through structure
abaqus_scripts.py could generate the snap-through structures and correspoding stretching simulation results.
The machine learning model is built by an inverse model and a forward model. The functions that define network structures are stored in utils.py and modules.py. The inverse_network_hypersearch.py and forward_network_hypersearch.py are designed to search the optimal parameters of the machine learning model. The inverse_network_batchsize.py and forward_network_batchsize.py are designed to search a suitable batchsize of the machine learning model. The inverse_network_best.py and forward_network_best.py are the final result of the machine learning model. Furthermore, the model result (.keras) are also attached.
To obtain the results in article again, we provide the train and test data.
