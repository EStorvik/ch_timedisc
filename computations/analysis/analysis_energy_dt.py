import numpy as np
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import NDArray


# input_folder = "log/accurate_dissipation_test/"
input_folder_e: str = "log/eyre_test/"
input_folder_ie: str = "log/implicit_euler_test/"


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # or 'sans-serif'
plt.rcParams["font.size"] = 16

line_styles: List[str] = ["-", "--", ":", "-."]

plt.figure()

for i in range(4):
    df = np.load(input_folder_ie + f"dt_{i}.npz")
    dt = df["dt"]
    energy = df["Energy"]
    time = df["Time"]
    plt.plot(
        time,
        energy,
        label=r"$\tau =$ " + f"{dt[0]:.0e}",
        linestyle=line_styles[i],
        linewidth=2,
    )


plt.legend()
plt.xlabel("Time / " + r"$t$")
plt.ylabel("Energy / " + r"$\mathcal{E(\varphi)}$")
plt.grid(True, alpha=0.8, linestyle=":", linewidth=0.5)
plt.savefig("energy_dt_implicit_euler.pdf", bbox_inches="tight", dpi=300)
plt.show()
