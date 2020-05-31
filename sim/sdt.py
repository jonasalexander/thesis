import numpy as np
import matplotlib.pyplot as plt

# Signal-detection expected gain for analytic solution
# based on the probability of Vh_1 coming from V as opposed to Vp
def SDT_gain(
    V=1,
    Vp=0,
    N=20,
    sigma=1,
    threshold=0.25,
    cost_eval=0.1,
    get_gain=False,
    get_dynamic=False,
):

    Vh_1_list = np.random.normal(V, sigma, N)  # draw N Vh_1 from V
    Vh_2_list = np.random.normal(Vp, sigma, N)  # draw N Vh_2 from V'

    probs = np.zeros(N)  # probability that Vh_1 is drawn from V (as opposed to V')
    gain = np.zeros(N)  # expected gain from evaluating A_1
    switch = np.zeros(N, dtype="int")  # whether Vh_1 < Vh_2
    default = np.zeros(N)  # value if choose action A_1 always
    dynamic = np.zeros(N)  # value if choose dynamically to evaluate or not
    did_eval = np.zeros(N, dtype="int")  # 1 if agent evaluated

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
        default[i] = [V, Vp][switch[i]]
        dynamic[i] = default[i]

        if gain[i] > threshold:
            dynamic[i] = V
            did_eval[i] = 1

    if get_gain:
        return gain
    elif get_dynamic:
        return dynamic - default - (did_eval * cost_eval)
    else:
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


def SDT_gain_fn(min_val, max_val, param, num=20, num_samples=1000, **kwargs):
    vals = np.linspace(min_val, max_val, num=num)
    gains = np.zeros(num)
    for i, s in enumerate(vals):
        kwargs.update({param: vals[i]})
        gains[i] = sum(SDT_gain(**kwargs, N=num_samples)) / num_samples

    return vals, gains


SDT_gain()
SDT_gain(V=0.1)
SDT_gain(V=2)
SDT_gain(V=11, Vp=10)
SDT_gain(sigma=2)

sigmas, gains = SDT_gain_fn(0.1, 10, "sigma", get_gain=True)
plt.plot(sigmas, gains)
plt.xlabel("Sigma")
plt.ylabel("Average gain")
plt.title("Average gain as a function of sigma for V=1, V'=0")
plt.show()

# average gain as a fn of V, fixing Vp at 0 and for 3 different values of sigma
for s in [0.2, 1, 2]:
    Vs, gains = SDT_gain_fn(
        0.1, 10, "V", get_gain=True, num=50, sigma=s, num_samples=10000
    )
    plt.plot(Vs, gains)
    plt.xlabel("V")
    plt.ylabel("Average gain")
    plt.title(f"Average gain as a function of V for sigma={s}, V'=0")
    plt.show()

for cost_eval in [0.1, 0.2, 0.3, 0.4, 0.5]:
    ts, rel_gains = SDT_gain_fn(
        0,
        0.5,
        "threshold",
        cost_eval=cost_eval,
        num=20,
        get_dynamic=True,
        num_samples=10000,
    )
    plt.plot(ts, rel_gains)
    plt.xlabel("Threshold")
    plt.ylabel("Average difference in dynamic vs default value")
    plt.title(
        f"Average advantage of dynamic over default "
        f"as a function of evaluation threshold for "
        f"evaluation cost={cost_eval}, V=1, V'=0, sigma=1"
    )
    plt.show()
