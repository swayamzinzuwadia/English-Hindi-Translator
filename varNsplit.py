from sklearn.model_selection import train_test_split


def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    return x_train, x_test, y_train, y_test
