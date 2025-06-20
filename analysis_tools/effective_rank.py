import numpy as np
import matplotlib.pyplot as plt
from libla import Matrix
from Signal_source.Model_signals import create_component_list, get_wavenums, create_coeff_list

wavenum = get_wavenums(10.0)

scheme = np.linspace(-40, 40, 20)

coeffs = create_coeff_list(5, 4, True, scheme, wavenum)

matrix = Matrix(coeffs)

def show(res):
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(20,5))
    def P(A): return np.abs(A)
    axs[0].imshow(P(matrix))
    axs[1].imshow(P(res.X))
    axs[2].imshow(P(res.R))
    axs[3].imshow(P(res.Y))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

#show(X.rank_decomposition(method="lu"))
#show(matrix.rank_decomposition(method="qr"))
show(matrix.rank_decomposition(method="svd"))
print(matrix.rank())

plt.show()