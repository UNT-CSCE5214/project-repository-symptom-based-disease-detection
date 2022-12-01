# making prediction by taking mode of predictions made by all the classifiers

svm_p = final_svm.predict(test_X)
nb_p = final_nbm.predict(test_X)
rf_p = final_rfm.predict(test_X)

final_p = [mode([i,j,k])[0][0] for i,j,k in zip(svm_p, nb_p, rf_p)]

# printing accuracies
print(f"Accuracy of combined model for Testing data: {accuracy_score(test_Y, final_p)*100}")