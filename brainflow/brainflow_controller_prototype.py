import argparse
import time

import matplotlib

matplotlib.use('gtk4Agg')
import matplotlib.pyplot as plt

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations


def plot_band(eeg_bands, colors, labels, sampling_rate):


    plt.xlim(0, sampling_rate)
    plt.ylim(0, 1)


    plt.xticks([0,  sampling_rate//2 , sampling_rate])
    plt.yticks([0, .5, 1])

    eeg_bands = np.array(eeg_bands) 
    pos = 0
    while pos < eeg_bands.shape[1]:        
        plt.plot(eeg_bands[: , pos],color=colors[pos], label=labels[pos])
        pos += 1

    plt.legend(loc="lower left")


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)
    board_descr = BoardShim.get_board_descr(args.board_id)
    sampling_rate = int(board_descr['sampling_rate'])   


    print("get sampling rate", sampling_rate)

    try:
        
        board.prepare_session()
        board.start_stream()

        past_bands = []


        time.sleep(3)
        while True:
            data = board.get_current_board_data(sampling_rate)  # get all data and remove it from internal buffer
            eeg_channels = board_descr['eeg_channels']


            bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
            feature_vector = bands[0]
            past_bands.append(feature_vector)

            if len(past_bands) <  sampling_rate:
                plot_band(past_bands, ["#FF0000","#DDA0DD","#00CED1","#556B2F","#FF8C00"], ["delta", "theta", "alpha","beta","gamma"],  sampling_rate)


            else:
                past_bands = past_bands[-sampling_rate:]
                plot_band(past_bands, ["#FF0000","#DDA0DD","#00CED1","#556B2F","#FF8C00"], ["delta", "theta", "alpha","beta","gamma"],  sampling_rate)

            plt.pause(0.01)
            plt.clf()


        board.stop_stream()
        board.release_session()

    except KeyboardInterrupt:
        np.savetxt("brain_wave_recording.csv", past_bands,delimiter=',')


if __name__ == "__main__":
    main()