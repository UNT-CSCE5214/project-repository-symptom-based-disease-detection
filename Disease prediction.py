import warnings
warnings.simplefilter('ignore')

sym = X.columns.values

sym_ind = {}
for index, value in enumerate(sym):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    sym_ind[symptom] = index

d_dic = {
    "sym_ind":sym_ind,
    "predictions_classes":enc.classes_
}

def Disease_Prediction(sym):
    sym = sym.split(",")
    
    # creating input data for the models
    input_data = [0] * len(d_dic["sym_ind"])
    for symptom in sym:
        index = d_dic["sym_ind"][symptom]
        input_data[index] = 1
		
		# reshaping the input data and converting it into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    rf_prediction = d_dic["predictions_classes"][final_rfm.predict(input_data)[0]]
    nb_prediction = d_dic["predictions_classes"][final_nbm.predict(input_data)[0]]
    svm_prediction = d_dic["predictions_classes"][final_svm.predict(input_data)[0]]
	
	
	   # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rfm_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
		}
		
		 # defining the global variable for storing the value of final_prediction and we are going to use it later
    global final_disease_predicted
    final_disease_predicted = final_prediction
    return predictions