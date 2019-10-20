import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

def get_returns(path_to_events_file):
    l = []
    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                l.append(v.simple_value)
    return l

def plot_lists(list_of_lists, name, mean=False):
    list_of_lists = [get_returns(x) for x in list_of_lists]
    for i, l in enumerate(list_of_lists):
        plt.plot(range(len(l)), l, label=str(i+1))
    if mean:
        meanlist = []
        for i in range(min(len(x) for x in list_of_lists)):
            s = sum([x[i] for x in list_of_lists])
            meanlist.append(float(s)/len(list_of_lists))
        plt.plot(range(len(l)), meanlist, label="Mean")
    plt.legend()
    plt.savefig(name)
    plt.clf()

if __name__ == "__main__":
    list_of_experiments_q1 = ["data/dqn_q1_PongNoFrameskip-v4_18-10-2019_02-30-19/events.out.tfevents.1571365819.floydhub"]
    # First Plot for Q1
    plot_lists(list_of_experiments_q1, "q1.png")

    # Plots for Q2
    list_of_experiments_q2 = ["data/dqn_q2_dqn_1_LunarLander-v2_17-10-2019_14-32-43/events.out.tfevents.1571347963.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_q2_dqn_2_LunarLander-v2_17-10-2019_20-47-53/events.out.tfevents.1571370473.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_q2_dqn_3_LunarLander-v2_17-10-2019_20-53-43/events.out.tfevents.1571370823.Abhisheks-MacBook-Pro-3.local"]
    plot_lists(list_of_experiments_q2, "q2.png", mean=True)

    # Plots for Q2 Double DQN
    list_of_experiments_q2p2 = ["data/dqn_double_q_q2_doubledqn_1_LunarLander-v2_17-10-2019_21-38-47/events.out.tfevents.1571373527.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_double_q_q2_doubledqn_2_LunarLander-v2_17-10-2019_21-39-24/events.out.tfevents.1571373564.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_double_q_q2_doubledqn_3_LunarLander-v2_17-10-2019_21-42-13/events.out.tfevents.1571373733.Abhisheks-MacBook-Pro-3.local"]
    plot_lists(list_of_experiments_q2p2, "q2p2.png", mean=True)

    # Plots for Q3
    list_of_experiments_q3 = ["data/dqn_q2_dqn_1_LunarLander-v2_17-10-2019_14-32-43/events.out.tfevents.1571347963.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_q3_hparam1_LunarLander-v2_17-10-2019_22-20-43/events.out.tfevents.1571376043.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_q3_hparam2_LunarLander-v2_17-10-2019_22-21-46/events.out.tfevents.1571376106.Abhisheks-MacBook-Pro-3.local",
        "data/dqn_q3_hparam3_LunarLander-v2_17-10-2019_22-25-52/events.out.tfevents.1571376352.Abhisheks-MacBook-Pro-3.local"]
    plot_lists(list_of_experiments_q3, "q3.png", mean=True)

    #Plots for Q4

    #Plots for Q5
