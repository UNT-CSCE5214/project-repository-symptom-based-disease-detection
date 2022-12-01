# training all the models on whole data

final_svm = SVC()
final_nbm = GaussianNB()
final_rfm = RandomForestClassifier(random_state = 18)
final_svm.fit(X, y)
final_nbm.fit(X, y)
final_rfm.fit(X, y)