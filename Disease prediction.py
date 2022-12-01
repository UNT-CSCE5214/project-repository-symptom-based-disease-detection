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