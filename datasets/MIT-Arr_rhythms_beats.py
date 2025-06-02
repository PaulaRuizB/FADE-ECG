import os
import wfdb
import sys
sys.path.append('../../')
import heartpy.filtering
from scipy import signal
import deepdish as dd
import numpy as np
import random

def generate_normal_images_train(train_complete_N, win, fs, record_name, valid_lead, counter):

    min_list = []
    max_list = []
    labels = []
    train_test_label = []

    for train_N in range(len(train_complete_N)):
        for i in range(int(train_complete_N[train_N][1]), int(train_complete_N[train_N][2]), fs):
            min_limit = i
            max_limit = i + win * fs

            if (max_limit <= int(train_complete_N[train_N][2])):
                maxi, mini, counter = create_image(min_limit, max_limit, record_name, valid_lead, counter)
                min_list.append(mini)
                max_list.append(maxi)
                labels.append("(N")
                train_test_label.append(1)  # TRAIN

    return min_list, max_list, counter, labels, train_test_label

def generate_normal_images_test(updated_rhythms, updated_beats, fs, record_name, valid_lead, counter):

    min_list = []
    max_list = []
    labels = []
    train_test_label = []
    complete_list_N = []

    for rhythm in range(len(updated_rhythms)):
        if updated_rhythms[rhythm][0] == "(N":
            for beat in range(len(updated_beats)):
                if updated_beats[beat][0] == "(N":
                    biggest = max(int(updated_beats[beat][1]), int(updated_rhythms[rhythm][1]))
                    smallest = min(int(updated_beats[beat][2]), int(updated_rhythms[rhythm][2]))

                    if (smallest - biggest) >= (fs * 5):
                        n_images = (smallest - biggest) // (fs * 5)
                        complete_list_N.append(("(N)", biggest, smallest, n_images))

    counter_N = sum(c[-1] for c in complete_list_N)
    test_idx = sorted(random.sample(range(1, counter_N), int(0.20 * counter_N)))

    train_complete_list_N = []

    init_c = 1
    for c in complete_list_N:
        min_window = int(c[1])
        max_window = int(c[1])
        continuous = 0
        for idx in range(init_c, init_c + c[-1]):
            if idx not in test_idx:
                max_window += 5 * fs
                continuous += 1
            else:
                if continuous > 0:
                    train_complete_list_N.append((c[0], min_window, max_window, continuous))
                min_window_test = max_window
                max_window_test = min_window_test + (5 * fs)
                maxi, mini, counter = create_image(min_window_test, max_window_test, record_name, valid_lead, counter)
                min_list.append(mini)
                max_list.append(maxi)
                labels.append("(N")
                train_test_label.append(0)  # TEST

                continuous = 0
                min_window = max_window_test
                max_window = min_window

        if continuous > 0:
            train_complete_list_N.append((c[0], min_window, max_window, continuous))

        init_c += c[-1]

    return min_list, max_list, counter, labels, train_test_label, train_complete_list_N

def same_change_in_both(changes1, changes2):

    updated_changes1 = []
    changes2_to_remove = []

    for change1 in changes1:
        in_both = False
        for change2 in changes2:
            if change1[0] == change2[0]:
                in_both = True
                updated_changes1.append((change1[0], change1[1] + change2[1], max(change1[2], change2[2])))
                changes2_to_remove.append(change2)
                continue

        if not in_both:
            updated_changes1.append(change1)

    for change in changes2_to_remove:
        changes2.remove(change)

    return updated_changes1, changes2
def crosscheck_validation(changes_list, opposite_list, fs, record_name, valid_lead, counter):

    min_list = []
    max_list = []
    labels = []
    train_test_label = []
    final_list = []

    for change in changes_list:
        for idx in range(len(opposite_list)):
            if  opposite_list[idx][0] == "(N" and (int(opposite_list[idx][1]) <= change[0] <= int(opposite_list[idx][2])) and (change[0]-int(opposite_list[idx][1])) >= 4*fs:
                maxi, mini, counter = create_image(change[0]-4*fs, change[0]+fs, record_name, valid_lead, counter)
                if change[0] - (4*fs) - int(opposite_list[idx][1])>=5*fs:
                    final_list.append(("(N", str(opposite_list[idx][1]), str(change[0]-(4*fs))))
                opposite_list[idx][1] = str(change[2])
                min_list.append(mini)
                max_list.append(maxi)
                labels.append("(N" + change[1])
                train_test_label.append(0)  # TEST

    if len(changes_list)>0:
        for op in range(len(opposite_list)):
            if (int(opposite_list[op][2])-int(opposite_list[op][1])>=5*fs) and opposite_list[op][0]=="(N":
                final_list.append(("(N", str(opposite_list[op][1]), str(opposite_list[op][2])))


    if len(final_list)==0:
        final_list = opposite_list

    return min_list, max_list, counter, labels, train_test_label, final_list
def valid_changes(complete_list, fs):

    changes = []
    for idx in range(len(complete_list)-1):
        if complete_list[idx][0] == "(N" and complete_list[idx+1][0] != "(N" and (int(complete_list[idx][2])-int(complete_list[idx][1]))>=4*fs:
            if ((int(complete_list[idx+1][2])-int(complete_list[idx+1][1]))<fs) and (idx+2)<len(complete_list):
                complete_list[idx+2][1] = str(int(complete_list[idx][2])+fs)

                if complete_list[idx + 2][0] != "(N":
                    changes.append((int(complete_list[idx][2]), complete_list[idx + 1][0] + complete_list[idx + 2][0], int(complete_list[idx][2])+fs))
                else:
                    changes.append((int(complete_list[idx][2]), complete_list[idx + 1][0] + complete_list[idx + 2][0], int(complete_list[idx][2])+fs))

            else:
                changes.append((int(complete_list[idx][2]), complete_list[idx+1][0], int(complete_list[idx+1][2])))

    return changes, complete_list

def generate_category_ranges(positions, labels):
    category_ranges = []
    current_category = None
    start_position = None

    for i in range(len(labels)):
        position = positions[i]
        label = labels[i].strip()

        if label:
            if current_category is None:
                current_category = label
                start_position = position
            elif current_category != label:
                # End of the previous category is the start of the next non-empty label
                end_position = positions[i] if labels[i] else positions[i - 1]
                category_ranges.append((current_category, start_position, end_position))
                current_category = label
                start_position = position

    # Handle the last category
    if current_category is not None:
        # If there are no more categories, set the end position to the last position in the array
        end_position = positions[-1]
        category_ranges.append((current_category, start_position, end_position))

    return np.array(category_ranges)

def create_image(min_limit, max_limit, record_name, valid_lead, counter):

    fs_new = 128
    win = 5
    #window
    window_complete = tmp_data[min_limit:max_limit]

    # Save path
    counter = counter + 1
    name_sample = "{}_{}_{:09d}".format(record_name, valid_lead[0], counter)
    names_samples.append(name_sample)

    # Median Filter
    med = signal.medfilt(window_complete, 3)

    # Bandpass Filter
    filt = heartpy.filtering.filter_signal(med, [0.5, 30], fs, order=2,
                                           filtertype='bandpass')

    # Resampling
    signal_new = signal.resample(filt, fs_new * win)

    # Re-segmentation
    win_size = 4
    signal_x = signal_new[:fs_new * win_size]
    signal_y = signal_new[fs_new:(fs_new * win_size) + fs_new]

    signal_xy = np.concatenate((np.expand_dims(signal_x, 1), np.expand_dims(signal_y, 1)), axis=1)
    signal_xy = np.transpose(signal_xy, (1, 0))

    dd.io.save(save_path + name_sample + '.h5', signal_xy)

    return signal_xy.max(), signal_xy.min(), counter

if __name__ == "__main__":

    path = '/mit-bih-arrhythmia-database-1.0.0/' # Path to the MIT-BIH Arrhythmia Database
    save_path = '/save_path/' # Path to save the processed dataset
    valid_lead = ['MLII'] #, 'II', 'I', 'MLI', 'V5'

    names_samples = []
    max_=[]
    min_=[]
    labels = []
    win = 5
    train_test_label = [] #0 TEST #1 TRAIN
    list_idxs = []

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        print(record_name)
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print('read data failed')
            continue

        fs = tmp_data_res[1]['fs']

        lead_in_data = tmp_data_res[1]['sig_name']
        my_lead_all = []


        #IF LEAD MLII is in this person
        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)

        #If there is a valid lead
        if len(my_lead_all) != 0:
            for my_lead in my_lead_all:

                #Data
                channel = lead_in_data.index(my_lead)
                tmp_data = tmp_data_res[0][:, channel]

                # Positions
                idx_list = list(tmp_ann_res['sample'])

                #Labels
                label_beat_list = tmp_ann_res['symbol']
                replacements = {'N': '(N', '/': '(N', '.': '(N', '[': '(N', '!': '(N', ']': '(N', 'x': '(N', '(': '(N', ')': '(N', 'p': '(N', 't': '(N', 'u': '(N', '`': '(N',
                                '\'': '(N', '^': '(N', '|': '(N', '~': '(N', '+': '(N', 's': '(N', 'T': '(N', '*': '(N', 'D': '(N', '=': '(N', '"': '(N', '@': '(N'}  # Define replacements
                label_beat_list = [replacements.get(x, x) for x in label_beat_list]

                label_rhythm = tmp_ann_res['aux_note']
                replacements_r = {'(P\x00': '(N'}  # Define replacements
                label_rhythm = [replacements_r.get(x, x) for x in label_rhythm]

                list_idxs.append(idx_list)
                #Rhythm and beats ranges
                rhythms = generate_category_ranges(np.array(idx_list), np.array(label_rhythm))
                beats = generate_category_ranges(np.array(idx_list), np.array(label_beat_list))

                counter = 0

                # VALID BEATS AND RHYTHMS
                changes_rhythms, updated_rhythms = valid_changes(rhythms, fs)
                changes_beats, updated_beats = valid_changes(beats, fs)

                # If change at the same time in both
                changes_rhythms, changes_beats = same_change_in_both(changes_rhythms, changes_beats)

                # Compare changes in rhythms with beats
                _min_list, _max_list, counter, _labels, _train_test_label, updated_beats = crosscheck_validation(changes_rhythms, updated_beats, fs, record_name, valid_lead, counter)
                min_.extend(_min_list)
                max_.extend(_max_list)
                labels.extend(_labels)
                train_test_label.extend(_train_test_label)

                # Compare changes in beats with rhythms
                _min_list, _max_list, counter, _labels, _train_test_label, updated_rhythms = crosscheck_validation(changes_beats, updated_rhythms, fs, record_name, valid_lead, counter)
                min_.extend(_min_list)
                max_.extend(_max_list)
                labels.extend(_labels)
                train_test_label.extend(_train_test_label)

                # N test images
                _min_list, _max_list, counter, _labels, _train_test_label, train_complete_N = generate_normal_images_test(updated_rhythms, updated_beats, fs, record_name, valid_lead, counter)
                min_.extend(_min_list)
                max_.extend(_max_list)
                labels.extend(_labels)
                train_test_label.extend(_train_test_label)

                # N train images
                _min_list, _max_list, counter, _labels, _train_test_label, = generate_normal_images_train(train_complete_N, win, fs, record_name, valid_lead, counter)
                min_.extend(_min_list)
                max_.extend(_max_list)
                labels.extend(_labels)
                train_test_label.extend(_train_test_label)

        else:
            print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
            continue


    information_dataset = {'total_samples': len(names_samples), 'labels': labels, 'list_names': names_samples,
                           'signal_description': np.array(("X_signal", "Y_signal")), 'max': max_, 'min':min_,
                           'train_test_label': train_test_label}

    dd.io.save(save_path + "mit_bih_arrhythmia_rhythm_beat_dataset_info.h5", information_dataset)