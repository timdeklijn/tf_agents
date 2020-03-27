from gridworld import GridWorldEnv

import tensorflow as tf
import matplotlib.pyplot as plt

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

replay_buffer_capacity = 100000
learning_rate = 1e-5
batch_size = 128
num_iterations = 10000
log_interval = 200
eval_interval = 1000
num_eval_episodes = 2


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


if __name__ == "__main__":
    print("Setting up environments")
    train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
    eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    print("Creating deep q_network")
    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("Creating Agent")
    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
    )
    tf_agent.train_step_counter.assign(0)

    tf_agent.initialize()
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    print("Creating reaply buffer")
    print("- Data Spec:", tf_agent.collect_data_spec._fields)
    print("- Batch Size:", train_env.batch_size)
    print("- Buffer Capacity", replay_buffer_capacity)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )
    replay_observer = [replay_buffer.add_batch]

    print("Create dataset form environment")
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    print("Setup training")
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    print("Setup driver, this will step through the environment")
    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=1,
    )

    print("Setup training loop")
    episode_len = []
    final_time_step, policy_state = driver.run()

    print("Start training loop")
    for i in range(num_iterations):
        final_time_step, policy_state = driver.run(final_time_step, policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f"step = {step}: loss = {train_loss.loss}")
            episode_len.append(train_metrics[3].result().numpy())

        if step % eval_interval == 0:
            avg_return = compute_avg_return(
                eval_env, tf_agent.policy, num_eval_episodes
            )
            print(f"step = {step}: Average Return = {avg_return}")

    plt.plot(episode_len)
    plt.savefig("result.png")
