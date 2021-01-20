import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform
import time


def main():
    X = np.matrix("26 16; 28 18; 31 21; 46 36; 47 37; 50 40; 51 41; 53 43")
    # a X b
    Y = np.matrix("1 -1; -1 1; 1 -1; -1 1; -1 1; 1 -1; -1 1; -1 1")
    # a X c

    mu = 0.001
    mu_max = 1000000
    rho = 1.2
    epsilon = 0.000001
    iter_max = 1000
    iter = 0
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    sigmak = 0.1

    # Get shapes of matriaaces
    a = X.shape[0] # number of training examples (n)
    b = X.shape[1] # number of features (d)
    c = Y.shape[1] # number of classes (c)

    Z = np.zeros((b, c))
    J = Z.copy()
    E = np.zeros((a, c))
    B = np.zeros((a, c))
    M1 = np.zeros((a, c))
    M2 = np.zeros((a, c))
    M3 = np.zeros((b, c))
    L = calc_L(X, sigmak) # a X a
    I = np.identity(b) # b X b

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
        A2 = B - X * J # a X c
        A3 = Z - J # b X c
        M1 = M1 + mu * A1 # a X c
        M2 = M2 + mu * A2 # a X c
        M3 = M3 + mu * A3 # b X c
        mu = min(rho * mu, mu_max)
        iter += 1
        print("Y - B - E = {}".format(la.norm(A1, "fro")))
        print("    Y - B = {}".format(la.norm(Y-B, "fro")))
        print("    Y - E = {}".format(la.norm(Y-E, "fro")))
        print("B - X * J = {}".format(la.norm(A2, "fro")))
        print("    Z - J = {}".format(la.norm(A3, "fro")))
        if iter > iter_max:
            converge = True
        if la.norm(A1, "fro") < epsilon and la.norm(A2, "fro") < epsilon and la.norm(A3, "fro") < epsilon:
            converge = True
        if iter > 500:
            break
    print("E")
    print(np.around(E))
    print("Y")
    print(np.around(Y, 2))
    print("T")
    print(np.around(X * Z))

def update_Z(J, Z, M3, mu, lambda1):
    tau_func = np.vectorize(lambda t: max(t-tau, 0))
    tau = 1.0 / (2.0 * lambda1 + mu)
    T_carot = tau * (mu * J - M3)
    U, S, V = la.svd(T_carot)
    S_carot = tau_func(np.diag(S))
    return U * S_carot * V


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
        M_norm[i] = la.norm(M_carot[i], 2)
    E = np.zeros(M_carot.shape)
    for i in range(num_rows):
        if M_norm[i] > eta:
            for j in range(num_cols):
                E[i, j] = M_carot[i, j] * (M_norm[i] - eta) / M_norm[i]
    return E


def update_J(B, I, L, X, Z, M2, M3, lambda2, mu):
    den = la.inv(2 * lambda2 * np.transpose(X) * L * X + mu * np.transpose(X) * X + mu * I)
    num = np.transpose(X) * M2 + M3 + mu * np.transpose(X) * B + mu * Z
    return den * num


def update_B(E, J, X, Y, M1, M2, mu):
    pi_func = np.vectorize(lambda t: min(max(t, -1), 1))
    B_carot = (mu * (Y - E + X * J) + M1 - M2) / (2 * mu)
    return pi_func(B_carot)


if __name__ == "__main__":
    main()
