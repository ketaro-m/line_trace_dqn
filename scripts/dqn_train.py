#! /usr/bin/env python

import rospy
import rosparam
import os
import json
import joblib
import numpy as np
import time
import argparse
import datetime
import pprint
from tkinter import filedialog
from collections import deque

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.line_trace_dqn.dqn_agent import DQNAgent
from src.line_trace_dqn.env import Env, STATE_SIZE

from std_msgs.msg import Float32MultiArray
import torch
import torch.nn.functional as F
import torch.optim as optim


LOG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log',
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))


def train(hyperparams: dict):
    rospy.init_node('turtlebot3_dqn_train')
    rosparam.set_param("/turtlebot3_dqn_train/log_path", LOG_PATH)
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    action_size = 5
    state_size = STATE_SIZE

    env = Env(action_size=action_size)
    agent = DQNAgent(state_size, action_size, hyperparams, seed=0)


    # 0. load saved model parameters
    if hyperparams['load_model']:
        typ = [('pth file','*.pth')]
        log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log')
        fle = filedialog.askopenfilename(filetypes = typ, initialdir = log_dir)
        agent.set_model_param(fle)
        hyperparams['load_model_file'] = fle

    pprint.pprint(agent.get_hyperparams())

    average_num = 10 #
    scores = [] # list containing score of each episode
    steps = []  # list containing steps of each episode
    average_scores = [] # average score of each n episodes
    average_steps = []  # average steps of each n episodes
    scores_window = deque(maxlen=average_num) # last 100 scores
    steps_window = deque(maxlen=average_num) # last 100 how many steps to be taken to goal


    # 0. save hyperparameters
    os.makedirs(LOG_PATH, exist_ok=True)
    with open(os.path.join(LOG_PATH , 'hyperparams.json'), 'w') as outfile:
        json.dump(hyperparams, outfile, indent=4)


    # 1. learning start
    while (not hasattr(env, 'image_raw')):
        rospy.loginfo('waiting for camera image...')
        time.sleep(0.5)
    rate = rospy.Rate(env.CONTROL_FREQ)
    rospy.loginfo('Learning start!')
    start_time = time.time()
    for i_episode in range(agent.load_episode+1, agent.n_episodes+1):
        done = False
        state = env.reset()
        score = 0
        step = 0
        time.sleep(1) # wait for steady

        # 1-1. try
        for t in range(1, agent.max_t+1):
            step = t
            action, q_values = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward * (agent.gamma ** t)
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # if failure (course out)
            if done:
                result.data = [score, step]
                pub_result.publish(result)
                break

            rate.sleep()


        scores_window.append(score) # save the most recent score
        scores.append(score)        # save the score
        steps_window.append(step)   # save the most recent step num
        steps.append(step)          # save the step num


        print('\rEpisode {}\t Epsilon {:.2f}\tScore {:.2f}\t@ {:0=3}\t'.format(i_episode,
                                                                                    agent.epsilon,
                                                                                    score,
                                                                                    step),
                                                                                    end="")

        # 1-2. save
        # 1-2-a. agent achieved max_t steps without failure : learning succeed
        if not done:
            scores.append(score)
            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            rospy.loginfo('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}\ttime: {:d}:{:02d}:{:02d}'.format(i_episode,
                                                                                        np.mean(scores_window),
                                                                                        h, m, s))
            param_keys = ['epsilon']
            param_values = [agent.epsilon]
            param_dictionary = dict(zip(param_keys, param_values))
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(LOG_PATH, 'endpoint_' + str(i_episode) + '.pth'))
            with open(os.path.join(LOG_PATH, 'param_' + str(i_episode) + '.json'), 'w') as outfile:
                json.dump(param_dictionary, outfile, indent=4)
            break


        # 1-2-b. every n episode, ROS log info, average
        if i_episode % average_num == 0:
            sys.stdout.write("\033[2K\033[G")
            sys.stdout.flush()
            print()
            rospy.loginfo('\rEpisode {}\t Average Score {:.4f} \tAgerage Step {:.2f}'.format(i_episode,
                                                                                            np.mean(scores_window),
                                                                                            np.mean(steps_window)))
            print()
            average_scores.append(np.mean(scores_window))
            average_steps.append(np.mean(steps_window))


        # 1-2-c.every 100 episode, save the model
        if i_episode % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(LOG_PATH, 'checkpoint_' + str(i_episode) + '.pth'))


        # 1-3. epsilon decay
        agent.update_epsilon()


    # 2. save the scores, steps
    joblib.dump((scores, average_scores, steps, average_steps, average_num),
                os.path.join(LOG_PATH, 'score_steps.txt'),
                compress=3)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## necessary to be executed from launch file: argv=[..., __name:=..., __log:=...]
    if any((s.startswith('__name') for s in sys.argv)):
        del sys.argv[-2:]
    ##

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--update_every', type=int, default=4)

    parser.add_argument('--n_episodes', type=int, default=3000)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.995)

    parser.add_argument('--load_model', type=bool, default=False)

    args = vars(parser.parse_args())

    try:
        train(hyperparams=args)
    except rospy.ROSInterruptException:
        pass

