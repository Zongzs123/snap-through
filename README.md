# snap-through
Machine learning method to inverse design a snap-through structure

Our FEA simulation is constructed utilizing ABAQUS 2023 with python scripts. Abaqus_scripts.py could generate the snap-through structures and correspoding stretching simulation results, and then export the results to txt files. 

The machine learning model is built by an inverse model and a forward model. The functions that define network structures are stored in utils.py and modules.py. The inverse_network_hypersearch.py and forward_network_hypersearch.py are designed to search the optimal parameters of the machine learning model. The inverse_network_batchsize.py and forward_network_batchsize.py are designed to search a suitable batchsize of the machine learning model. The inverse_network_best.py and forward_network_best.py are the final result of the machine learning model. Furthermore, the model result (.keras) are also attached. eval_demo.py is utilized to generate the results of two demonstrations. eval_inverse_design.py is utilized in inverse design with some certain parameters.

Furthermore, we report a forward model stored in forward_model.keras and six inverse models in inverse_model_0.keras, inverse_model_1.keras, inverse_model_3.keras, inverse_model_4.keras, inverse_model_6.keras, inverse_model_8.keras.

To obtain the results in article again, we provide the train and test data, "train.xlsx" and "test.xlsx".

The version of all python codes is 3.9.2. The version of keras and tensorflow is 2.10. The version of sklearn is 1.5.0.
