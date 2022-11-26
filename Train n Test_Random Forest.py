rfm = RandomForestClassifier(random_state=18)
rfm.fit(X_train, y_train)
p = rfm.predict(X_test)
