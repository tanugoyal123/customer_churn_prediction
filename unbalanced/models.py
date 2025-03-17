from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# defining function to train,evaluate logistic regression model
def logistic_regression(x_train,y_train,x_test,y_test):
    logreg = LogisticRegression().fit(x_train, y_train)
    y_pred_train = logreg.predict(x_train)
    y_pred_test = logreg.predict(x_test)
    cv_scores = cross_val_score(logreg, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train=accuracy_score(y_train,y_pred_train)
    c_report=classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d={"training_accuracy":accuracy_train,"testing_accuracy":accuracy_test,"classification_report":c_report,"model":logreg,"cross_val_score":cv_mean}
    return d 

# defining function to train,evaluate KNN model
def knn_classifier(x_train, y_train, x_test, y_test, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_train, y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)
    cv_scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    c_report = classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d = {"training_accuracy": accuracy_train,
         "testing_accuracy": accuracy_test,
         "classification_report": c_report,
         "model": knn,
         "cross_val_score":cv_mean}
    return d

# defining function to train,evaluate decision tree model
def decision_tree(x_train, y_train, x_test, y_test):
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    y_pred_train = dt.predict(x_train)
    y_pred_test = dt.predict(x_test)
    cv_scores = cross_val_score(dt, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    c_report = classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d = {"training_accuracy": accuracy_train,
         "testing_accuracy": accuracy_test,
         "classification_report": c_report,
         "model": dt,"cross_val_score":cv_mean}
    return d

# defining function to train,evaluate random forest model
def random_forest(x_train, y_train, x_test, y_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42).fit(x_train, y_train)
    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)
    cv_scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    c_report = classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d = {"training_accuracy": accuracy_train,
         "testing_accuracy": accuracy_test,
         "classification_report": c_report,
         "model": rf,"cross_val_score":cv_mean}
    return d


# defining function to train,evaluate SVM model
def svm_classifier(x_train, y_train, x_test, y_test, kernel='linear'):
    svm = SVC(kernel=kernel).fit(x_train, y_train)
    y_pred_train = svm.predict(x_train)
    y_pred_test = svm.predict(x_test)
    cv_scores = cross_val_score(svm, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    c_report = classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d = {"training_accuracy": accuracy_train,
         "testing_accuracy": accuracy_test,
         "classification_report": c_report,
         "model": svm,"cross_val_score":cv_mean}
    return d

# defining function to train,evaluate naive bayes model
def naive_bayes(x_train, y_train, x_test, y_test):
    nb = GaussianNB().fit(x_train, y_train)
    y_pred_train = nb.predict(x_train)
    y_pred_test = nb.predict(x_test)
    cv_scores = cross_val_score(nb, x_train, y_train, cv=5, scoring='accuracy')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    c_report = classification_report(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    d = {"training_accuracy": accuracy_train,
         "testing_accuracy": accuracy_test,
         "classification_report": c_report,
         "model": nb,"cross_val_score":cv_mean}
    return d

