# reading the test data

test_data = pd.read_csv("/content/Testing(1).csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = enc.transform(test_data.iloc[:, -1])