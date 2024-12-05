import gymnasium as gym

import argparse
import time

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations


def plot_band(eeg_bands, colors, labels, sampling_rate):

    plt.xlim(0, sampling_rate)
    plt.ylim(0, 1)

    plt.xticks([0, sampling_rate // 2, sampling_rate])
    plt.yticks([0, 0.5, 1])

    eeg_bands = np.array(eeg_bands)
    pos = 0
    while pos < eeg_bands.shape[1]:
        plt.plot(eeg_bands[:, pos], color=colors[pos], label=labels[pos])
        pos += 1

    plt.legend(loc="lower left")


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board_id = 39

    board = BoardShim(board_id, params)
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = 1024

    print("get sampling rate", sampling_rate)

    try:

        board.prepare_session()
        board.start_stream()

        env = gym.make("MountainCar-v0", render_mode="human")
        observation, info = env.reset()

        past_bands = []

        time.sleep(3)

        episode_over = False
        while not episode_over:
            data = board.get_current_board_data(
                sampling_rate
            )  # get all data and remove it from internal buffer
            eeg_channels = board_descr["eeg_channels"]

            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            eeg_band_vector = bands[0]
            past_bands.append(eeg_band_vector)

            if len(past_bands) < sampling_rate:
                plot_band(
                    past_bands,
                    ["#FF0000", "#DDA0DD", "#00CED1", "#556B2F", "#FF8C00"],
                    ["delta", "theta", "alpha", "beta", "gamma"],
                    sampling_rate,
                )

            else:
                past_bands = past_bands[-sampling_rate:]
                plot_band(
                    past_bands,
                    ["#FF0000", "#DDA0DD", "#00CED1", "#556B2F", "#FF8C00"],
                    ["delta", "theta", "alpha", "beta", "gamma"],
                    sampling_rate,
                )

            plt.pause(0.01)
            plt.clf()

            if eeg_band_vector[0] > 0.5:
                action = 2
            else:
                action = 0

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated

        env.close()
        board.stop_stream()
        board.release_session()

    except KeyboardInterrupt:
        np.savetxt("brain_wave_recording.csv", past_bands, delimiter=",")
        env.close()


if __name__ == "__main__":
    main()
