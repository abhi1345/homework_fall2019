import matplotlib.pyplot as plt

def get_returns(path_to_events_file):
    l = []
    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                l.append(v.simple_value)
    return l

def plot_lists(list_of_lists):
    for i, l in enumerate(list_of_lists):
        plt.plot(range(len(l)), l, label=str(i))
    plt.show()

if __name__ == "__main__":
    list_of_experiments_cartpole = ["data/pg_sb_no_rtg_dsa_CartPole-v0_01-10-2019_21-10-02/events.out.tfevents.1569989402.Abhisheks-MacBook-Pro-3.local", \
        "data/pg_sb_rtg_dsa_CartPole-v0_01-10-2019_21-15-02/events.out.tfevents.1569989702.Abhisheks-MacBook-Pro-3.local", \
        "data/pg_todo_CartPole-v0_01-10-2019_21-15-30/events.out.tfevents.1569989730.Abhisheks-MacBook-Pro-3.local", \
        "data/pg_lb_no_rtg_dsa_CartPole-v0_01-10-2019_21-17-31/events.out.tfevents.1569989851.Abhisheks-MacBook-Pro-3.local", \
        "data/pg_lb_rtg_dsa_CartPole-v0_01-10-2019_21-18-00/events.out.tfevents.1569989880.Abhisheks-MacBook-Pro-3.local", \
        "data/pg_todo_CartPole-v0_01-10-2019_21-18-19/events.out.tfevents.1569989899.Abhisheks-MacBook-Pro-3.local"]
    # First Plot for CartPole
    plot_lists(list_of_experiments_cartpole[:2])

    # Second Plot for CartPole
    plot_lists(list_of_experiments_cartpole[2:])

    # Inverted Pendulum Plot
    list_of_inverted_pendulum = ["data/pg_ip_b1000_r0.01_InvertedPendulum-v2_03-10-2019_19-01-03/events.out.tfevents.1570129263.floydhub"]
    plot_lists(list_of_inverted_pendulum)
