#! /usr/bin/env python

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


def main():
    LOG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log')
    fld = filedialog.askdirectory(initialdir = LOG_PATH)
    fname = 'score_steps.txt'
    fpath = os.path.join(fld, fname)

    scores, average_scores, steps, average_steps, average_num = joblib.load(fpath)

    fig, axes = plt.subplots(2, 1, figsize=(12,8))
    # score
    axes[0].plot(np.arange(len(scores)), scores)
    axes[0].plot(np.arange(average_num//2, average_num*len(average_scores), average_num), average_scores)
    axes[0].set_xlabel('Episode #')
    axes[0].set_ylabel('Score')
    # step num
    axes[1].plot(np.arange(len(steps)), steps)
    axes[1].plot(np.arange(average_num//2, average_num*len(average_steps), average_num), average_steps)
    axes[1].set_xlabel('Episode #')
    axes[1].set_ylabel('Steps')
    # save
    fig.savefig(os.path.join(fld, 'score_steps.png'))
    plt.show()




if __name__ == '__main__':
    main()