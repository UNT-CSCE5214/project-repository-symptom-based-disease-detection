sym_list = list()
sym_list.append(symptom1)
sym_list.append(symptom2)
sym_list.append(symptom3)
sym_list_string = str(",".join(sym_list))
print(sym_list_string)
# print(",".join(sym_list))

print((Disease_Prediction(sym_list_string)))