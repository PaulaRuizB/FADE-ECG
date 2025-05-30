def name_save_path(save_path_train, prefix, loss1, optim_name,
                   batch_size, lr, min_lr, dropout, sqi, num_channels, total_epochs,
                   ft, weights_mse_inorout_lastsecond, threshold_mse_inside_outside,
                   add_FC, norm_a_b_max_min,
                   dataset_FT, input_size_seconds):

    save_path_train = save_path_train + prefix + "_" +  loss1 + "_" + optim_name + \
                              '_bs{:04d}_lr{:0.7f}_sched_warmup0_min_lr{:0.13f}_dr{:0.2f}_sqi{:0.2f}_channels{:02d}_epochs{:03d}'.format(
                                  batch_size, lr, min_lr, dropout, sqi, num_channels, total_epochs)

    if ft:
        save_path_train = save_path_train + "_ft"

    if loss1 == 'mse_inside_outside_thresholds' or loss1 == "mse_last_second_separate" or loss1 == "mse_inside_outside_thresholds_times_max" or loss1 == "median_ae_inside_outside_thresholds" or loss1 == "mse_last_second_separate_times_max":
        save_path_train = save_path_train + "_w_" + str(weights_mse_inorout_lastsecond[0]) + "_" + str(
            weights_mse_inorout_lastsecond[1])

    if loss1 == "mse_last_second_separate_in_out":
        save_path_train = save_path_train + "_w_" + str(weights_mse_inorout_lastsecond[0]) + "_" + str(
            weights_mse_inorout_lastsecond[1]) + "_" + str(weights_mse_inorout_lastsecond[2])

    if loss1 == 'mse_inside_outside_thresholds' or loss1 == "mse_inside_outside_thresholds_times_max" or loss1 == "median_ae_inside_outside_thresholds" or loss1 == "triple_band_mse" or loss1 == "mse_last_second_separate_in_out":
        save_path_train = save_path_train + "_th_" + str(threshold_mse_inside_outside[0]) + "_" + str(
            threshold_mse_inside_outside[1])

    if add_FC:
        save_path_train = save_path_train + "_FC"

    if norm_a_b_max_min:
        save_path_train = save_path_train + "_norm_a_b_max_min"

    if ft:
        save_path_train = save_path_train + "_{}".format(dataset_FT)

    save_path_train = save_path_train + "_{}s".format(input_size_seconds)

    return save_path_train
