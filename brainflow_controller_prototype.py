import argparse
import time
import matplotlib


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from tensorflow.keras.models import load_model  # tensorflow 2.13.1 for the fix

# Path to your trained model
CLASSICAL_RNN_MODEL_PATH = "models/classic_rnn_model.h5"


def parse_arguments():
    """
    Parse command-line arguments for board configuration.

    Returns:
        argparse.Namespace: Parsed arguments for the board connection.
    """
    parser = argparse.ArgumentParser(
        description="Real-time EEG visualization with BrainFlow."
    )
    parser.add_argument(
        "--board-id", type=int, required=True, help="Board ID for the EEG device."
    )
    parser.add_argument(
        "--serial-port", type=str, default="", help="Serial port (if applicable)."
    )
    parser.add_argument(
        "--ip-address", type=str, default="", help="IP address (if applicable)."
    )
    parser.add_argument(
        "--mac-address", type=str, default="", help="MAC address (if applicable)."
    )
    parser.add_argument(
        "--ip-port", type=int, default=0, help="IP port (if applicable)."
    )
    parser.add_argument(
        "--timeout", type=int, default=0, help="Timeout for device discovery."
    )
    parser.add_argument(
        "--file", type=str, default="", help="File path for playback (if applicable)."
    )
    return parser.parse_args()


def load_trained_model(model_path):
    """
    Load a trained model from the specified path.

    Parameters:
    - model_path (str): Path to the saved model file.

    Returns:
    - model: The loaded Keras model if successful, otherwise None.
    """
    try:
        model = load_model(model_path)

        if model is None:
            raise ValueError(
                f"Failed to load Classic Model from {CLASSICAL_RNN_MODEL_PATH}"
            )

        print("Model loaded successfully.")

        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def setup_board(params, board_id):
    """
    Initialize and configure the EEG board.

    Args:
        params (BrainFlowInputParams): Configuration parameters for BrainFlow.
        board_id (int): Board ID for the EEG device.

    Returns:
        tuple: Board instance, sampling rate, and EEG channel list.
    """
    board = BoardShim(board_id, params)
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = board_descr["sampling_rate"]
    eeg_channels = board_descr["eeg_channels"]
    return board, sampling_rate, eeg_channels


def preprocess_streaming_data(data, eeg_channels, sampling_rate, window_size=256):
    """
    Extracts average band powers for the specified window and normalizes them.

    Args:
        data: np.ndarray, raw EEG data from the Muse headband.
        eeg_channels: list, indices of EEG channels to process.
        sampling_rate: int, sampling rate of the EEG device.
        window_size: int, number of samples in the sliding window.

    Returns:
        np.ndarray: Normalized band power feature vector of shape
                    (time_steps, num_features).
    """
    # Extract the most recent window of data
    # Shape: (num_channels, window_size)
    windowed_data = data[:, -window_size:]

    # Compute average band powers
    bands = DataFilter.get_avg_band_powers(
        windowed_data, eeg_channels, sampling_rate, apply_filter=True
    )
    avg_band_powers = bands[0]  # Shape: (num_features,)

    # Normalize band powers
    normalized_band_powers = (avg_band_powers - np.mean(avg_band_powers)) / np.std(
        avg_band_powers
    )

    return normalized_band_powers


def plot_band(eeg_bands, colors, labels, sampling_rate):
    """
    Plot EEG band powers with real-time updates.

    Args:
        eeg_bands (list): List of EEG band power vectors.
        colors (list): Colors for the band power lines.
        labels (list): Labels for the band power lines.
        sampling_rate (int): Sampling rate of the EEG device.
    """
    plt.xlim(0, sampling_rate)
    plt.ylim(0, 1)
    plt.xticks([0, sampling_rate // 2, sampling_rate])
    plt.yticks([0, 0.5, 1])

    eeg_bands = np.array(eeg_bands)
    for idx, band in enumerate(eeg_bands.T):
        plt.plot(band, color=colors[idx], label=labels[idx])

    plt.legend(loc="lower left")
    plt.xlabel("Time")
    plt.ylabel("Normalized Band Power")


def main():
    """
    Main function for streaming, processing, and visualizing EEG data
    in real time.
    """
    # Enable logging for BrainFlow
    BoardShim.enable_dev_board_logger()

    # Parse arguments and configure BrainFlow parameters
    args = parse_arguments()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    params.ip_address = args.ip_address
    params.mac_address = args.mac_address
    params.ip_port = args.ip_port
    params.timeout = args.timeout
    params.file = args.file

    # Setup EEG board
    board, sampling_rate, eeg_channels = setup_board(params, args.board_id)

    # Load the trained model
    classic_model = load_trained_model(CLASSICAL_RNN_MODEL_PATH)

    try:
        # Start the EEG session
        board.prepare_session()
        board.start_stream()
        print(
            f"""Streaming EEG data from board ID {args.board_id} at
            {sampling_rate} Hz..."""
        )

        # Initialize data buffer
        # past_bands = []
        # plt.figure()

        # Initialize rolling prediction buffer
        predictions = []

        while True:
            # Get current data from the board
            data = board.get_current_board_data(sampling_rate)
            if data.shape[1] < sampling_rate:
                # print("Data shape:", data.shape)
                continue

            # Preprocess the data for model input
            feature_vector = preprocess_streaming_data(
                data, eeg_channels, sampling_rate
            )
            feature_vector = feature_vector[
                np.newaxis, :
            ]  # Expand dimensions to match model input shape (1, 20)

            # Predict using models
            classic_pred = classic_model.predict(feature_vector)

            # Convert probabilities to class labels
            classic_class = np.argmax(classic_pred)

            # Aggregate predictions for visualization
            predictions.append((classic_class))

            # Maintain a rolling window for visualization
            if len(predictions) > sampling_rate // 2:
                predictions = predictions[-(sampling_rate // 2) :]

            # Plot real-time predictions
            plt.plot([p[0] for p in predictions], label="Classic Model", color="blue")
            plt.plot(
                [p[1] for p in predictions], label="Extended Fuzzy Model", color="green"
            )
            plt.plot(
                [p[2] for p in predictions], label="Fuzzy Layer Model", color="red"
            )
            plt.xlabel("Time")
            plt.ylabel("Predicted Class")
            plt.legend()
            plt.pause(0.01)
            plt.clf()

            # Get current data from the board
            # data = board.get_current_board_data(sampling_rate)
            # if data.shape[1] < sampling_rate:
            #    print("Data shape:", data.shape)
            #    continue
            # Compute average band powers
            # bands = DataFilter.get_avg_band_powers(
            #    data, eeg_channels, sampling_rate, apply_filter=True
            # )
            #
            # feature_vector = bands[0]  # Only average powers
            # past_bands.append(feature_vector)

            # Maintain a rolling window for visualization
            # if len(past_bands) > sampling_rate:
            #    past_bands = past_bands[-sampling_rate:]

            # Plot the band powers
            # plot_band(
            #    past_bands,
            #    colors=["#FF0000", "#DDA0DD", "#00CED1", "#556B2F", "#FF8C00"],
            #    labels=["Delta", "Theta", "Alpha", "Beta", "Gamma"],
            #    sampling_rate=sampling_rate,
            # )
            # plt.pause(0.01)
            # plt.clf()

    except KeyboardInterrupt:
        print("Streaming interrupted. Saving data to 'brain_wave_recording.csv'...")
        # np.savetxt("brain_wave_recording.csv", past_bands, delimiter=",")
    finally:
        # Stop and release the EEG session
        board.stop_stream()
        board.release_session()
        print("Session ended.")


if __name__ == "__main__":
    main()
