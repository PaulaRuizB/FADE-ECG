import torch.nn.functional as F

def mse_inside_outside_thresholds(y_gt, y_pred, lower_threshold, upper_threshold, reduction_mse):
    # Apply the thresholds
    mask_between = (y_gt >= lower_threshold) & (y_gt <= upper_threshold)
    mask_outside = ~mask_between

    # Values between the thresholds
    y_gt_between = y_gt[mask_between]
    y_pred_between = y_pred[mask_between]

    # Values outside the thresholds
    y_gt_outside = y_gt[mask_outside]
    y_pred_outside = y_pred[mask_outside]

    # Calculate MSE for values between the thresholds
    mse_between = F.mse_loss(y_pred_between, y_gt_between, reduction=reduction_mse)

    # Calculate MSE for values outside the thresholds
    mse_outside = F.mse_loss(y_pred_outside, y_gt_outside, reduction=reduction_mse)

    return mse_between, mse_outside
