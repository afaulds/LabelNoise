import numpy as np
from scipy.spatial.distance import pdist, squareform


def main():
    X = np.matrix("26 16; 28 18; 31 21; 46 36; 47 37; 50 40; 51 41; 53 43")
    # a X b
    Y = np.matrix("1 -1; -1 1; 1 -1; -1 1; -1 1; 1 -1; -1 1; -1 1")
    # c X b

    mu = 0.001
    mu_max = 1000000
    rho = 1.2
    epsilon = 0.000001
    iter_max = 1000
    iter = 0
    lambda1 = 0.1 #FIXME
    lambda2 = 0.1 #FIXME
    lambda3 = 0.1 #FIXME

    # Get shapes of matriaaces
    a = X.shape[0] # number of features
    b = X.shape[1] # number of training examples
    c = Y.shape[0] # number of classes

    Z = np.zeros((c, a))
    J = Z.copy()
    E = np.zeros((c, b))
    B = np.zeros((c, b))
    M1 = np.zeros((c, b))
    M2 = np.zeros((c, b))
    M3 = np.zeros((c, a))

    pi_func = np.vectorize(lambda t: min(max(t, -1), 1))

    converge = False
    while not converge:
        L = equ3(X)
        T = ??
        W = REPLACE
        Z = equ10(T)
        E = REPLACE
        J = np.linalg.inv(2 * lambda2 * np.transpose(X) * L * X)

        B_carot = (mu * (Y - E + np.matmul(J, X)) + M1 - M2) / (2 * mu)
        B = pi_func(B_carot)

        A1 = Y - B - E #(c x b)
        A2 = B - np.matmul(J, X) #(c x b)
        A3 = Z - J #()
        M1 = M1 + mu * A1 #(c x b)
        M2 = M2 + mu * A2 #()
        M3 = M3 + mu * A3 #()
        mu = min(rho * mu, mu_max)
        iter += 1
        if iter < iter_max:
            converge = True
        if norm(A1) < epsilon:
            converage = True
        if norm(A2) < epsilon:
            converage = True
        if norm(A3) < epsilon:
            converage = True


def equ3(X):
    s = 1
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    W_carot = np.exp(-pairwise_dists ** 2 / (2 * s ** 2))
    D = np.zeros(W_carot.shape)
    D = np.diag(np.sum(W_carot, axis=0))
    print(D)
    return D - W_carot


def equ10(T):
    u, s, v = np.linalg.svd(T_carot)

if __name__ == "__main__":
    main()
