import os
import torch
import deepdish as dd
from scipy import signal
import heartpy.filtering
import numpy as np
import sys
import wfdb
sys.path.append('../')
from ecg_qc.ecg_qc import EcgQc
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

def main(part: int):

    database_path = "/mit-bih-normal-sinus-rhythm-database-1.0.0/" # Path to the MIT-BIH Normal Sinus Rhythm Database
    data_files = [database_path + file for file in os.listdir(database_path) if ".dat" in file]
    save_path = '/save_path/' # Path to save the processed dataset

    names_samples = []
    fs = 128
    win = 5
    fs_new = 128
    signal_quality = []

    counters = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                16: 0, 17: 0}

    ## Loop ecg.bin files
    for participant, file in enumerate(data_files):
        counter = 0

        if participant==part:

            logging.critical("Participant: " + str(participant) + "/" + str(len(data_files)))

            # Get signal
            data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
            data["Participant"] = "MIT-Normal_%.2i" %(participant)
            data["Sample"] = range(len(data))
            data["Sampling_Rate"] = 128
            data["Database"] = "MIT-Normal"

            data = data['ECG'].values
            ECG_SIGNAL = torch.Tensor(data)  # Final numpy array containing full days record

            # windows of 10 s
            for i in range(0, data.size, fs):
                min_limit = i
                max_limit = i + win * fs

                if (max_limit <= data.size):

                    # Save path
                    counter = counter + 1
                    name_sample = "{}_{:012d}".format(participant, counter)
                    names_samples.append(name_sample)

                    window_complete = ECG_SIGNAL[min_limit:max_limit]

                    # Median Filter
                    med = signal.medfilt(window_complete, 3)

                    # Bandpass Filter
                    filt = heartpy.filtering.filter_signal(med, [0.5, 30], fs, order=2,
                                                           filtertype='bandpass')

                    # Signal Quality
                    signal_quality_now = []
                    win_intra = 1
                    for k in range(win_intra * fs, window_complete.shape[0] + 1, win_intra * fs):
                        min_limit = k - win_intra * fs
                        max_limit = k
                        window_complete_intra = filt[min_limit:max_limit]
                        ecg_qc = EcgQc(model_file='rfc_2s.pkl', sampling_frequency=fs, normalized=False)
                        signal_quality_now.append(ecg_qc.get_signal_quality(window_complete_intra))

                    signal_quality.append(np.mean(signal_quality_now))

                    # Re-segmentation
                    win_size = 4
                    signal_x = filt[:fs_new * win_size]
                    signal_y = filt[fs_new:(fs_new * win_size)+fs_new]

                    signal_xy =np.concatenate((np.expand_dims(signal_x, 1), np.expand_dims(signal_y, 1)), axis=1)
                    signal_xy = np.transpose(signal_xy, (1, 0))

                    dd.io.save(save_path + name_sample + '.h5',
                               signal_xy)
                    counters[participant] += 1


    shape_samples = signal_xy.shape

    information_dataset = {'participant': part,'images_size': shape_samples, 'total_samples': len(names_samples),
                            'list_names': names_samples,
                            'signal_description': np.array(("X_signal", "Y_signal")),
                            'signal_quality': signal_quality}

    dd.io.save(save_path + "mit_bih_nsr_info_complete_{}.h5".format(part), information_dataset)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='images')
    parser.add_argument('--participant', default=1, type=int, help='')

    args = parser.parse_args()
    participant = args.participant
    print("Finishing participant: ", participant)

    main(participant)