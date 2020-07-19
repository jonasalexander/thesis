import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_agent_data(agents, normalize=True, limit=50):
    """Plot utility of different agents for each trial.

    Parameters
    ----------
    agents : list of agents that extend BaseDecisionMaker. Must all have the
        same env (env of the first agent)
    limit : integer, optional
        Number of trials for which to display utility. Defaults to 50.
    plot_num_eval : Boolean, optional
        Whether to show the number of actions the dynamic agent evaluates
        on another y axis. Defaults to True.
    """

    if normalize:
        agents_data = [agent.normalize_util().loc[:limit-1] for agent in agents]
    else:
        agents_data = [agent.data.loc[:limit-1] for agent in agents]

    to_plot = pd.DataFrame(columns=["index", "agent", "variable", "value"])
    for i, agent in enumerate(agents):
        agent_data = agents_data[i]
        to_plot = to_plot.append(agent_data.reset_index().melt(id_vars="index").assign(agent=agent.name))

    to_plot["index"] = to_plot["index"].astype("int")
    to_plot["value"] = to_plot["value"].astype("float")

    g = sns.relplot(
        kind="line",
        data=to_plot,
        x="index",
        y="value",
        hue="agent",
        col="variable",
    )
    g.fig.suptitle("Agent behavior for agents " + ", ".join([agent.name for agent in agents]))
    return g

def plot_distribution_comparison(dynamic_agent, fixed_agent, datatype):
    """For the dynamic and fixed agents given, compare the distribution over
    datatype in trials where num_eval was identical for the dynamic agent.
    """
    trials = dynamic_agent.data.loc[dynamic_agent.data["num_eval"] == fixed_agent.num_eval]
    dynamic_values = trials[datatype].value_counts(normalize=True)
    fixed_values = fixed_agent.data[datatype].value_counts(normalize=True)

    domain = set(fixed_values.index)
    domain.update(dynamic_values.index)
    domain = sorted(list(domain))

    ind = np.arange(len(domain))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, [dynamic_values[x]*100 if x in dynamic_values else 0 for x in domain], width,
                    label='dynamic')
    rects2 = ax.bar(ind + width/2, [fixed_values[x]*100 if x in fixed_values else 0 for x in domain], width,
                    label=fixed_agent.name)

    ax.set_ylabel(datatype + ' %')
    ax.set_title('Distribution over ' + datatype + ' in trials where num_eval was identical for the dynamic agent (n=' + str(len(trials)) + ')')
    ax.set_xticks(ind)
    ax.set_xticklabels([x+1 for x in domain])
    ax.legend()

    fig.tight_layout()
    plt.show()
