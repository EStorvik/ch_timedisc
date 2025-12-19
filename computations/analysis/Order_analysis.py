import numpy as np

re8 = np.load("results_em8.npz")
re7 = np.load("results_IE_em7.npz")
re6 = np.load("results_IE_em6.npz")
re5 = np.load("results_IE_em5.npz")
re4 = np.load("results_IE_em4.npz")


pf_8 = re8["pf"]
pf_7 = re7["pf"]
pf_6 = re6["pf"]
pf_5 = re5["pf"]
pf_4 = re4["pf"]

dt8, _, _ = re8["dt, ell, nx"]
dt7, _, _ = re7["dt, ell, nx"]
dt6, _, _ = re6["dt, ell, nx"]
dt5, _, _ = re5["dt, ell, nx"]
dt4, _, _ = re4["dt, ell, nx"]

t8 = re8["Time"]
t7 = re7["Time"]
t6 = re6["Time"]
t5 = re5["Time"]
t4 = re4["Time"]


def L2(p1, p2, dt2, t1, t2, n):
    e = 0
    tf = 0
    for i in range(len(p2)):
        e += np.inner(p1[n * i] - p2[i], p1[n * i] - p2[i]) * dt2
        tf += abs(t1[n * i] - t2[i])

    print(tf)
    return e


f1 = L2(pf_8, pf_5, dt5, t8, t5, 1000)
f2 = L2(pf_8, pf_6, dt6, t8, t6, 500)
f3 = L2(pf_8, pf_7, dt7, t8, t7, 250)
f4 = L2(pf_8, pf_4, dt4, t8, t4, 125)

print(f1 / f2)
print(f2 / f3)
print(f3 / f4)
