import numpy as np
import matplotlib.pyplot as plt

# Signal-detection expected gain for analytic solution
# based on the probability of Vh_1 coming from V as opposed to Vp
def SDT_gain(V=1, Vp=0, N=20, sigma=1, plot=True, get_gain=False):

    Vh_1_list = np.random.normal(V, sigma, N)  # draw N Vh_1 from V
    Vh_2_list = np.random.normal(Vp, sigma, N)  # draw N Vh_2 from V'

    probs = np.ones(N)  # probability that Vh_1 is drawn from V (as opposed to V')
    gain = np.ones(N)  # expected gain from evaluating A_1
    switch = np.zeros(N)  # whether Vh_1 < Vh_2

    for i in range(N):

        Vh_1 = Vh_1_list[i]
        Vh_2 = Vh_2_list[i]

        if Vh_1 < Vh_2:
            switch[i] = 1
            Vh_1, Vh_2 = Vh_2, Vh_1  # switch so that Vh_1 is larger

        p1 = np.exp(-0.5 * (((Vh_1 - V) / sigma) ** 2 + ((Vh_2 - Vp) / sigma) ** 2))
        p2 = np.exp(-0.5 * (((Vh_1 - Vp) / sigma) ** 2 + ((Vh_2 - V) / sigma) ** 2))

        denom = p1 + p2
        probs[i] = p1 / denom
        gain[i] = V - (V * p1 + Vp * p2) / denom

    if get_gain:
        return gain
    elif plot:
        fig, ax1 = plt.subplots()
        fig.suptitle(f"V={V}, V'={Vp}, sigma={sigma}")

        ax1.plot(gain, label="expected gain from eval", color="r")
        ax1.set_ylabel("Expected gain from eval")

        diffs = abs(Vh_1_list - Vh_2_list)
        ax2 = ax1.twinx()
        ax2.plot(diffs, label="diff between vhats", color="b")
        ax2.set_ylabel("Difference between vhats")
        fig.legend()
        plt.show()


SDT_gain()
SDT_gain(V=0.1)
SDT_gain(V=2)
SDT_gain(V=11, Vp=10)
SDT_gain(sigma=2)

# average gain as a fn of sigma
num = 20
sigmas = np.linspace(0.1, 10, num=num)
gains = np.zeros(num)
for i, s in enumerate(sigmas):
    num_samples = 1000
    gains[i] = sum(SDT_gain(sigma=s, get_gain=True, N=num_samples)) / num_samples

plt.plot(sigmas, gains)
plt.xlabel("Sigma")
plt.ylabel("Average gain")
plt.title("Average gain as a function of sigma for V=1, V'=0")
plt.show()

# average gain as a fn of V, fixing Vp at 0
num = 50
Vs = np.linspace(0.1, 10, num=num)
gains = np.zeros(num)
for i, v in enumerate(Vs):
    num_samples = 10000
    gains[i] = sum(SDT_gain(V=v, get_gain=True, N=num_samples)) / num_samples

plt.plot(Vs, gains)
plt.xlabel("V")
plt.ylabel("Average gain")
plt.title("Average gain as a function of V for sigma=1, V'=0")
plt.show()

# average gain as a fn of V, fixing Vp at 0
num = 50
Vs = np.linspace(0.1, 10, num=num)
gains = np.zeros(num)
for i, v in enumerate(Vs):
    num_samples = 10000
    gains[i] = sum(SDT_gain(V=v, sigma=2, get_gain=True, N=num_samples)) / num_samples

plt.plot(Vs, gains)
plt.xlabel("V")
plt.ylabel("Average gain")
plt.title("Average gain as a function of V for sigma=2, V'=0")
plt.show()

# average gain as a fn of V, fixing Vp at 0
num = 50
Vs = np.linspace(0.1, 10, num=num)
gains = np.zeros(num)
for i, v in enumerate(Vs):
    num_samples = 10000
    gains[i] = sum(SDT_gain(V=v, sigma=0.2, get_gain=True, N=num_samples)) / num_samples

plt.plot(Vs, gains)
plt.xlabel("V")
plt.ylabel("Average gain")
plt.title("Average gain as a function of V for sigma=0.2, V'=0")
plt.show()
