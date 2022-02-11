import os
import librosa
import librosa.display
from librosa.filters import mel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import json

mel_spec_frame_size = 1024
n_mels = 32 # was 128

# smaller n_mels meant more data meant more batch size

def get_classes_in_data_path(datapath='./corpus'):
    subfolders = [f.path for f in os.scandir(datapath) if f.is_dir()]
    class_labels = np.arange(0,len(subfolders))

    return subfolders, class_labels


def get_wav_files_in_path(datapath):
    files = os.listdir(datapath)
    files_wav = [i for i in files if i.endswith('.wav')]

    return files_wav


def save_image(filepath, fig=None):
    """
    Save the current image with no whitespace to the given file path
    :param filepath: File path of PNG file to create
    :param fig: The matplotlib figure to save
    :return:
    """
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches=0, bbox_inches='tight')


def get_mel_spectrogram(wavfile, do_plot=False):
    """
    Given a path to a wav file, returns a Mel-spectrogram array,
    np.ndarray [shape=(n_mels,t)]
    :param wavfile: The input wav file path
    :param do_plot: Flag to either plot the spectrogram or not (for debugging)
    :return: Returns a tuple of np.ndarray [shape=(n_mels,t)] and fs
    """ 
    # load the wav file
    # sig is the audio signal
    # fs is the sampling frequency
    sig, fs = librosa.load(wavfile, sr=None)
    # keep the original signal shape
    sig_shape = sig.shape
    # normalize the time-domain signal to a range between -1.0 and 1.0 using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    sig = scaler.fit_transform(sig.reshape(-1, 1))
    # reshape sig back to sig_shape
    sig = np.reshape(sig, sig_shape)
    # compute the mel spectrogram
    # mel_spec is the mel spectrogram array
    mel_spec = librosa.feature.melspectrogram(  # since we are using librosa and we do not have a spectogram, we will pass: 
                                                y=sig, # the signal 
                                                sr=fs, # the sampling frequency
                                                n_fft=mel_spec_frame_size, # the number of samples per frame
                                                hop_length=mel_spec_frame_size//2, # the number of samples between each window 
                                                n_mels=n_mels # the number of mel filters
                                            )
    # power_to_db converts the mel spectrogram to decibels
    mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
    # plot the spectrogram
    if do_plot:
        # https://librosa.org/doc/main/auto_examples/plot_display.html
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', ax=ax)
        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        # save the image
        fig.savefig('mel_spec_test', pad_inches=0, bbox_inches='tight')
        plt.show()
        print(mel_spec.shape)
    # return the mel spectrogram and the sampling frequency
    return mel_spec, fs

def dataset_generation():
    # Get the list of folders in the corpus
    subfolders, _ = get_classes_in_data_path()
    # Get the list of wav files in each folder
    for folder in tqdm(subfolders, desc='Processing'):
        wav_files = get_wav_files_in_path(folder)
        delete_flag = True
        for each_file in wav_files:
            wav_file = os.path.join(folder, each_file)

            # create a corresponding folder in the image_dataset folder
            # (will only run once)
            image_folder = os.path.join('./image_dataset', folder)
            if not os.path.exists(image_folder):
                    os.makedirs(image_folder)

            # this might be a re-run, and I might have changed te mel_spec variable
            # so remove all the files in the folder
            for file in os.listdir(image_folder):
                if delete_flag:
                    os.remove(os.path.join(image_folder, file))

            # Get the mel spectrogram for each wav file
            mel_spec, _ = get_mel_spectrogram(wav_file)

            # mel_spec is a np.ndarray [shape=(n_mels,t)]
            # chunk mel_spec into arrays of size (n_mels, n_mels)
            for i in range(mel_spec.shape[1]//n_mels):
                # get the chunk of size (n_mels, n_mels)
                chunk = mel_spec[:, i*n_mels:(i+1)*n_mels]
                # generate a heatmap for the chunk using imshow
                fig = plt.figure()
                plt.imshow(chunk) 
                # save the chunk as a png file
                save_image(os.path.join(image_folder, f"{each_file.split('.')[0]}_{str(i)}.png"))
                # close the figure
                plt.close(fig)
            # flag (so these chunks don't get deleted)
            delete_flag = False

def json_to_text(json_file):
    '''
    small helper function to help with the json file to text file conversion
    '''
    with open(json_file) as f:
        data = json.load(f)
        # for each key in the json file, save the list (value) as a text file
        for key in data.keys():
            # save the list as a text file
            with open(f"{key}_paths.txt", "w") as f:
                for item in data[key]:
                    f.write(f"{item}\n")


if __name__ == '__main__':
    # # Test the get_mel_spectrogram function
    # get_mel_spectrogram('./corpus/ajh001/shortpassagea_CT.wav', do_plot=True)
    # # now run the dataset_generation function
    # dataset_generation()
    # # run the json_to_text function (Part 2 deliverable)
    json_to_text('Part 2/paths.json')
    pass