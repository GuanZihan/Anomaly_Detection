import numpy as np
import matplotlib.pyplot as plt

losses_clean = np.load("losses_clean.npy", allow_pickle=True)
losses_bad = np.load("losses_bad.npy", allow_pickle=True)

plt.plot(losses_clean)
plt.plot(losses_bad)

plt.show()