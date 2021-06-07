from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def decision_tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini', random_state=1)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class prediction: {(y_pred != y_test).sum()}\n")


def random_forest(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=25, criterion='gini', n_jobs=4, random_state=1)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    print(f"Random forest accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class prediction: {(y_pred != y_test).sum()}\n")


def svm_classifier(x_train, x_test, y_train, y_test):
    svc = SVC(C=10.0, kernel='rbf', random_state=1, gamma=0.1)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print(f"SVN accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class prediction: {(y_pred != y_test).sum()}\n")


def knn_classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(f"KNN accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Bad class prediction: {(y_pred != y_test).sum()}\n")