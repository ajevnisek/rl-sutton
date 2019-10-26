"""Fast implementation for 10 armed nonstationary bandit simulation."""
from os.path import isfile

import json
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_PATH = "."
RUN_PARAMETERS_PATH = "{}/parameters/non_stationary_bandit_run_parameters.json".format(BASE_PATH)
ARMED_BANDIT_RESULT = "results/run_results.json"

NON_STATIONARY_ARMED_BANDIT_RESULT = \
    '{}/results/fast_non_stationary_armed_bandit_run_results.json'.format(BASE_PATH)
RESULT_FIGURES_PATH = '{}/results/fast-10-armed-nonstationary-bandit-{}.png'

def epsilon_greedy_k_armed_non_stationary_bandit(k=10, epsilon=0.1, alpha=0.1,
                                                 independent_runs=2000,
                                                 horizon=10000):
    """Simulate epsilon greedy algorithm for k armed nonstationary bandit."""
    average_reward = numpy.zeros(horizon)
    success_rate = numpy.zeros(horizon)

    for run in xrange(independent_runs):
        q_star = numpy.zeros(k)
        q_t = numpy.zeros(k)
        # first sample all the random variables and then run simulation.
        random_walk = numpy.random.normal(loc=0, scale=0.01, size=(horizon, k))
        is_exploration_per_time_step = numpy.random.binomial(1, epsilon, size=horizon)
        random_arm_for_exploration = numpy.random.randint(0, k, size=horizon)
        noisy_reward = numpy.random.randn(horizon)

        for time_step in xrange(horizon):
            q_star += random_walk[time_step]
            best_arm = q_star.argmax()
            is_exploration = is_exploration_per_time_step[time_step]
            if is_exploration:
                selected_arm = random_arm_for_exploration[time_step]
            else:
                selected_arm = q_t.argmax()
            reward = q_star[selected_arm] + noisy_reward[time_step]
            q_t[selected_arm] = q_t[selected_arm] + alpha *(reward - q_t[selected_arm])
            average_reward[time_step] += reward
            success_rate[time_step] += (selected_arm == best_arm)
        if (run + 1) % 100 == 0:
            print "Finished run number #{}".format(run + 1)

    average_reward /= independent_runs
    success_rate /= independent_runs
    return average_reward, success_rate

def main():
    """Run all simulations."""
    with open('{}'.format(RUN_PARAMETERS_PATH), 'r') as js_file:
        json_data = js_file.read()
    run_parameters = json.loads(json_data)
    run_results = {x:{} for x in run_parameters.keys()}

    for run, parameters in run_parameters.items():
        print "Simulating run : {}".format(run)
        average_reward, success_rate = epsilon_greedy_k_armed_non_stationary_bandit(
            k=parameters['k'], epsilon=parameters['epsilon'])
        run_results[run]['Average Reward'] = list(average_reward)
        run_results[run]['Success Rate'] = list(success_rate)

    # plot results
    plot_metric(run_results, metric_name="Average Reward")
    plot_metric(run_results, metric_name="Success Rate")

    # log results
    with open(NON_STATIONARY_ARMED_BANDIT_RESULT, 'w') as js_file:
        js_file.write(json.dumps(run_results, indent=4, sort_keys=True))


def plot_metric(run_results, metric_name):
    """Plots a result metric vs the step number."""
    plt.figure()
    plt.title('{} vs Step - Non-Stationary Armed Bandit'.format(metric_name))
    plt.ylabel(metric_name)
    plt.xlabel('Steps')
    colors = 'rgb'
    for index, run in enumerate(run_results):
        plt.plot(run_results[run][metric_name], colors[index % len(colors)], label=run)
    if isfile(ARMED_BANDIT_RESULT):
        with open(ARMED_BANDIT_RESULT) as js_file:
            json_data = js_file.read()
        stationary_armed_bandit_results = json.loads(json_data)
        plt.plot(stationary_armed_bandit_results['eps=0.1'][metric_name], 'k', label="stationary")

    plt.legend(loc='upper right')
    plt.savefig(RESULT_FIGURES_PATH.format(BASE_PATH, metric_name))
    plt.close()


if __name__ == "__main__":
    main()
