rfm = RandomForestClassifier(random_state=18)
rfm.fit(X_train, y_train)
p = rfm.predict(X_test)
cfm = confusion_matrix(y_test, p)
plt.figure(figsize=(12,8))
sns.heatmap(cfm,cmap="YlGnBu", annot=True)
plt.title("Random Forest Classifier Confusion Matrix")
plt.show() 
#printing accuracies
print(f"Accuracy of Random Forest for training data: {accuracy_score(y_train, rfm.predict(X_train))*100}")

print(f"Accuracy of Random Forest for testing data: {accuracy_score(y_test, p)*100}")
