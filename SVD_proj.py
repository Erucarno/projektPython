import numpy as np
import unittest

def svd_decomposition(A):
    # Obliczanie A^T A oraz wartości i wektorów własnych
    ATA = np.dot(A.T, A)
    eigenvalues_v, V = np.linalg.eigh(ATA)

    # Sortowanie wartości i wektorów własnych w porządku malejącym
    sorted_indices = np.argsort(eigenvalues_v)[::-1]
    eigenvalues_v = eigenvalues_v[sorted_indices]
    V = V[:, sorted_indices]

    # Obliczanie wartości singularnych
    singular_values = np.sqrt(np.maximum(eigenvalues_v, 0))
    min_dim = min(A.shape)

    # Sigma jako macierz diagonalna
    Sigma = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    np.fill_diagonal(Sigma, singular_values[:min_dim])

    # Obliczanie A A^T oraz jego wartości i wektorów własnych
    AAT = np.dot(A, A.T)
    eigenvalues_u, U = np.linalg.eigh(AAT)

    # Sortowanie wartości i wektorów własnych w porządku malejącym
    sorted_indices_u = np.argsort(eigenvalues_u)[::-1]
    eigenvalues_u = eigenvalues_u[sorted_indices_u]
    U = U[:, sorted_indices_u]

    # Dostosowanie znaków U, aby pasowały do V
    for i in range(min_dim):
        if np.dot(np.dot(A, V[:, i]), U[:, i]) < 0:
            U[:, i] *= -1

    return U, Sigma, V.T

class TestSVD(unittest.TestCase):
    def test_svd_decomposition(self):
        A = np.array([
            [1, 0, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0]
        ])

        # Obliczanie oczekiwanego SVD
        expected_U, expected_Sigma_values, expected_VT = np.linalg.svd(A)
        expected_Sigma = np.zeros((A.shape[0], A.shape[1]), dtype=float)
        np.fill_diagonal(expected_Sigma, expected_Sigma_values)

        print("Oczekiwana macierz U:\n", expected_U)
        print("\nOczekiwana maciecz Sigma:\n", expected_Sigma)
        print("\nOczekiwana macierz VT:\n", expected_VT)

        # Obliczanie SVD
        U, Sigma, VT = svd_decomposition(A)

        print("\nMacierz U:\n", U)
        print("\nMacierz Sigma:\n", Sigma)
        print("\nMacierz VT:\n", VT)

        # Funkcja asercji do rozstrzygania niejednoznaczności znaku
        def assert_svd_almost_equal(U1, U2, Sigma1, Sigma2, V1, V2, decimal=3):
            for i in range(U1.shape[1]):
                if np.allclose(U1[:, i], -U2[:, i], atol=10**-decimal):
                    U2[:, i] *= -1
                    V2[i, :] *= -1
            np.testing.assert_almost_equal(U1, U2, decimal=decimal)
            np.testing.assert_almost_equal(Sigma1, Sigma2, decimal=decimal)
            np.testing.assert_almost_equal(V1, V2, decimal=decimal)

        assert_svd_almost_equal(U, expected_U, Sigma, expected_Sigma, VT, expected_VT)

if __name__ == '__main__':
    unittest.main()
