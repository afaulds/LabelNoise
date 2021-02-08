import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pickle
from scipy.spatial.distance import pdist, squareform
import time


def main():
    with open("data/Simple.pkl", "rb") as infile:
        data = pickle.loads(infile.read())

    X = data["X"]
    # a X b
    Y = data["Y"]
    # a X c

    mu = 0.001
    mu_max = 1000000
    rho = 1.1
    epsilon = 0.000001
    iter_max = 1000
    iter = 0
    lambda1 = 0.1
    lambda2 = 0.1
    lambda3 = 0.1
    sigmak = 0.1

    # Get shapes of matriaaces
    a = X.shape[0] # number of training examples (n)
    b = X.shape[1] # number of features (d)
    c = Y.shape[1] # number of classes (c)

    Z = np.zeros((b, c))
    J = np.zeros((b, c))
    E = np.zeros((a, c))
    B = np.zeros((a, c))
    M1 = np.zeros((a, c))
    M2 = np.zeros((a, c))
    M3 = np.zeros((b, c))
    L = calc_L(X, sigmak) # a X a
    I = np.identity(b) # b X b

    i = []
    a1_norm = []
    a2_norm = []
    a3_norm = []
    z = []

    converge = False
    while not converge:
        print("")
        print("---------------------------------------------------------")
        print("{}".format(iter))

        #Z_new = update_Z(J, Z, M3, mu, lambda1)
        #E_new = update_E(B, Y, M1, mu, lambda3)
        #J_new = update_J(B, I, L, X, Z, M2, M3, lambda2, mu)
        #B_new = update_B(E, J, X, Y, M1, M2, mu)

        #Z = Z_new.copy()
        #E = E_new.copy()
        #J = J_new.copy()
        #B = B_new.copy()

        Z = update_Z(J, Z, M3, mu, lambda1)
        E = update_E(B, Y, M1, mu, lambda3)
        J = update_J(B, I, L, X, Z, M2, M3, lambda2, mu)
        B = update_B(E, J, X, Y, M1, M2, mu)

        A1 = Y - B - E # a X c
        A2 = B - X @ J     # a X c
        A3 = Z - J # b X c
        M1 = M1 + mu * A1 # a X c
        M2 = M2 + mu * A2 # a X c
        M3 = M3 + mu * A3 # b X c
        mu = min(rho * mu, mu_max)
        iter += 1
        print("Y - B - E = {}".format(frobenius_norm(A1)))
        print("B - X * J = {}".format(frobenius_norm(A2)))
        print("    Z - J = {}".format(frobenius_norm(A3)))
        i.append(iter)
        a1_norm.append(frobenius_norm(A1))
        a2_norm.append(frobenius_norm(A2))
        a3_norm.append(frobenius_norm(A3))
        print(np.transpose(X @ J) @ L @ (X @ J))
        z.append(
            la.norm(Z, "nuc") +
            lambda1 * pow(la.norm(Z, "fro"), 2) +
            lambda3 * la.norm(E, 2) +
            pow(la.norm(Y - X @ Z - E, "fro"), 2)
        )
        if iter > iter_max:
            converge = True
        if frobenius_norm(A1) < epsilon and frobenius_norm(A2) < epsilon and frobenius_norm(A3) < epsilon:
            converge = True
    print("E")
    print(np.around(E, 2))
    print("Y")
    print(np.around(Y, 2))
    print("T")
    print(np.around(np.matmul(X, Z), 2))
    print(np.around(Y - E, 2))
    plt.plot(i, z)
    #plt.plot(i, a1_norm, "r-", i, a2_norm, "b-", i, a3_norm, "g-")
    plt.show()

def update_Z(J, Z, M3, mu, lambda1):
    tau = 1.0 / (2.0 * lambda1 + mu)
    T_carot = tau * (mu * J - M3)
    U, S, V = la.svd(T_carot)
    S_carot = np.zeros((U.shape[1], V.shape[0]))
    for i in range(min(U.shape[1], V.shape[0])):
        S_carot[i, i] = S[i]

    x = np.matmul(np.matmul(U, S_carot), V)
    if not np.allclose(T_carot, x):
        exit()

    for i in range(min(U.shape[1], V.shape[0])):
        S_carot[i, i] = max(S_carot[i, i] - tau, 0)

    return np.matmul(np.matmul(U, S_carot), V)


def calc_L(X, sigmak):
    pairwise_dists = squareform(pdist(X, "euclidean"))
    W_carot = np.exp(-pairwise_dists ** 2 / (2 * sigmak ** 2))

    D = np.zeros(W_carot.shape)
    D = np.diag(np.sum(W_carot, axis=0))
    return D - W_carot


def update_E(B, Y, M1, mu, lambda3):
    M_carot = Y - B + M1 / mu
    eta = lambda3 / mu

    num_rows = M_carot.shape[0]
    num_cols = M_carot.shape[1]

    M_norm = np.zeros(num_rows)
    for i in range(num_rows):
        M_norm[i] = l21_norm(M_carot[i])
    E = np.zeros(M_carot.shape)
    for i in range(num_rows):
        if M_norm[i] > eta:
            for j in range(num_cols):
                E[i, j] = M_carot[i, j] * (M_norm[i] - eta) / M_norm[i]
    return E


def update_J(B, I, L, X, Z, M2, M3, lambda2, mu):
    den = la.inv(2 * lambda2 * np.matmul(np.matmul(np.transpose(X), L), X) + mu * np.matmul(np.transpose(X), X) + mu * I)
    num = np.matmul(np.transpose(X), M2) + M3 + mu * np.matmul(np.transpose(X), B) + mu * Z
    return np.matmul(den, num)


def update_B(E, J, X, Y, M1, M2, mu):
    pi_func = np.vectorize(lambda t: min(max(t, -1), 1))
    B_carot = (mu * (Y - E + np.matmul(X, J)) + M1 - M2) / (2 * mu)
    return pi_func(B_carot)


def frobenius_norm(X):
    return la.norm(X, "fro")

def l21_norm(X):
    return la.norm(X, 2)

if __name__ == "__main__":
    main()
