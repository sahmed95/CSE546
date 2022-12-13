import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    lam = np.linspace(0, 50, 6)
    for i in range(len(lam)):
        model = PolynomialRegression(degree=d, reg_lambda=lam[i])
        model.fit(X, y)

    # output predictions
        xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
        ypoints = model.predict(xpoints)

        # plot curve
        plt.plot(X, y, "rx")
        plt.title(f"PolyRegression with d = {d}")
        plt.plot(xpoints, ypoints, label = f"$\lambda$ = {int(lam[i])}")
        plt.xlabel("X")
        plt.ylabel("Y")
    plt.legend(fontsize= 8, loc = "upper left")
    plt.show()
