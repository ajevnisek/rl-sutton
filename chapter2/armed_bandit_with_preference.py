"""10 armed bandit simulation."""
import json
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_PATH = "/mnt/c/Users/ajevn/Desktop/interview/rl"


def epsilon_greedy_k_armed_bandit(k=10, epsilon=0.1, independent_runs=2000,
                                  horizon=1000):
    """Simulate epsilon greedy algorithm for k armed bandit."""
    average_reward = numpy.zeros(horizon)
    success_rate = numpy.zeros(horizon)

    for run in xrange(independent_runs):
        q_star = numpy.random.randn(k)
        best_arm = q_star.argmax()

        q_t = numpy.zeros(k)
        times_visited_arm = numpy.zeros(k)
        h_t = numpy.zeros(k)
        for time_step in xrange(horizon):
            pi = numpy.exp(h_t)
            pi /= sum(pi)
            selected_arm = numpy.random.choice(range(k), p=pi)
            reward = q_star[selected_arm] + numpy.random.randn(1)
            h_t[selected_arm] = h_t[selected_arm] + alpha *(reward - average_reward) * (1 - pi[selected_arm])
            all_arms_but_selected_arm = range(k)
            all_arms_but_selected_arm.remove(selected_arm)
            h_t[all_arms_but_selected_arm] = h_t[all_arms_but_selected_arm] - alpha *(reward - average_reward)
            times_visited_arm[selected_arm] += 1
            q_t[selected_arm] = q_t[selected_arm] + \
                1.0 / times_visited_arm[selected_arm] *(reward - q_t[selected_arm])
            average_reward[time_step] += reward
            success_rate[time_step] += (selected_arm == best_arm)
        if (run + 1) % 100 == 0:
            print "Finished run number #{}".format(run + 1)

    average_reward /= independent_runs
    success_rate /= independent_runs
    return average_reward, success_rate

def main():
    """Run all simulations."""
    with open('{}/run_parameters.json'.format(BASE_PATH), 'r') as js_file:
        json_data = js_file.read()
    run_parameters = json.loads(json_data)
    run_results = {x:{} for x in run_parameters.keys()}

    for run, parameters in run_parameters.items():
        print "Simulating run : {}".format(run)
        average_reward, success_rate = epsilon_greedy_k_armed_bandit(
            k=parameters['k'], epsilon=parameters['epsilon'])
        run_results[run]['Average Reward'] = list(average_reward)
        run_results[run]['Success Rate'] = list(success_rate)

    # plot results
    plot_metric(run_results, metric_name="Average Reward")
    plot_metric(run_results, metric_name="Success Rate")

    # log results
    with open('{}/run_results.json'.format(BASE_PATH), 'w') as js_file:
        js_file.write(json.dumps(run_results, indent=4, sort_keys=True))


def plot_metric(run_results, metric_name):
    """Plots a result metric vs the step number."""
    plt.figure()
    plt.title('{} vs Step'.format(metric_name))
    plt.ylabel(metric_name)
    plt.xlabel('Steps')
    colors = 'rgb'
    for index, run in enumerate(run_results):
        plt.plot(run_results[run][metric_name], colors[index % len(colors)], label=run)
    plt.legend(loc='upper right')
    plt.savefig('{}/10-armed-bandit-{}.png'.format(BASE_PATH, metric_name))
    plt.close()


if __name__ == "__main__":
    main()
