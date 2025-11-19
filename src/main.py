from feature_extraction import prepare_features
from naive_bayes_manual import ManualGaussianNB
from naive_bayes_sklearn import train_sklearn_gaussian_nb
from evaluation import evaluate_model, print_summary_table

def main():
    X_train_50, X_test_50, y_train, y_test = prepare_features()

    results = []

    # Naive Bayes from Scratch
    nb_manual = ManualGaussianNB()
    nb_manual.fit(X_train_50, y_train)
    y_pred = nb_manual.predict(X_test_50)

    acc, prec, rec, f1, cm = evaluate_model(
        "Naive Bayes (manual)", y_test, y_pred
    )
    results.append(("Naive Bayes (manual)", acc, prec, rec, f1))

    # Naive Bayes Sklearn
    nb_sk = train_sklearn_gaussian_nb(X_train_50, y_train)
    y_pred_sk = nb_sk.predict(X_test_50)

    acc, prec, rec, f1, cm = evaluate_model(
        "Naive Bayes (sklearn)", y_test, y_pred_sk
    )
    results.append(("Naive Bayes (sklearn)", acc, prec, rec, f1))

    print_summary_table(results)

if __name__ == "__main__":
    main()
