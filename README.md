# FADE-ECG

*FADE: Forecasting for Anomaly Detection on ECG*

### Prerequisites
1. Clone this repository with git:
```
git clone https://github.com/PaulaRuizB/FADE-ECG
```
2. What you need to use the codes:
   
   * For training and testing: Python 3.10.6 and requirements fade_requirements.txt into venv_requirements folder
   * For inference on NVIDIA Jetson Orin Nano: TensorRT 10.3.0 and cuda 12.5. 

3. Datasets:
   * MIT-BIH Normal Sinus Rhythm (NSR) database: https://physionet.org/content/nsrdb/1.0.0/
   * MIT-BIH Arrhythmia database: https://physionet.org/content/mitdb/1.0.0/

4. Process the datasets:
   * MIT-BIH Normal Sinus Rhythm (NSR) database: run ```datasets/MIT_BIH_normal_dataset.py``` after specifying the ```database_path``` (location of the raw dataset) and ```save_path``` (destination for the processed data) arguments.

     _Note: Signal quality index for the NSR dataset has been calculated using: [ECG_QC library](https://github.com/Aura-healthcare/ecg_qc)
     
   * MIT-BIH Arrhythmia database: run ```datasets/MIT-Arr_rhythms_beats.py``` after specifying the ```path``` (location of the raw dataset) and ```save_path``` (destination for the processed data) arguments.
   
### Train baseline forecasting model with PyTorch:
```
python -m torch.distributed.run --nproc_per_node=4 ../mains/train_original.py --total_epochs 200 --save_every 25 --batch_size 256 --optim_name adamw --lr 1e-4 --min_lr 1e-7 --dropout 0.2 --loss1 mse_inside_outside_thresholds --weights_mse_inorout_lastsecond 5.0 1.0 --threshold_mse_inside_outside 0.4 -0.4 --sqi 0.5 --num_channels 1 --norm_a_b_max_min --input_size_seconds 4 --add_FC --prefix repo --path_train_set /path_train_set1/ --path_datasetinfo /path_dataset_info/mit_bih_nsr_info_complete.h5 --save_path_train /save_path/
```

### Train domain-adapted model with PyTorch:
```
python -m torch.distributed.run --nproc_per_node=4 ../mains/train_original.py --total_epochs 250 --save_every 25 --batch_size 64 --optim_name adamw --lr 1e-4 --min_lr 1e-8 --dropout 0.2 --loss1 mse_inside_outside_thresholds --ft --weights_mse_inorout_lastsecond 1.0 1.0 --threshold_mse_inside_outside 0.3 -0.3 --sqi -1 --num_channels 1 --dataset_FT mit_arr_beats --input_size_seconds 4 --add_FC --prefix repo --path_pretrained_model /path_pretrained_model/ --path_train_set /path_train_set2/ --path_datasetinfo /path_dataset_info/mit_bih_arrhythmia_rhythm_beat_dataset_info.h5 --save_path_train /save_path/
```

### Test forecasting models with PyTorch:
```
python ../mains/test_threshold_and_acc.py --dropout 0 --num_channels 1 --add_FC --path_trained_model /path_pretrained_model/ --path_datasetinfo /path_dataset_info/mit_bih_arrhythmia_rhythm_beat_dataset_info.h5 --path_test_set /path_test_set/ --path_save_results /save_path/ --step_percentile 0.0001 --test_percentage 0.2 --seed 23 --input_size_seconds 4
```

### Optimize models with TensorRT for inference:
From PyTorch to ONNX (utils folder):
```
python3 torch_to_onnx.py --weights /path_model/
```
From ONNX to TensorRT
* FP32 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/
```
* FP16 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --fp16
```
* INT8 GPU
```
/usr/src/tensorrt/bin/trtexec --onnx=/path_onnx_model/ --saveEngine=/path_save_trt/ --int8
```

### Our paper: [FADE: Forecasting for anomaly detection on ECG](https://doi.org/10.1016/j.cmpb.2025.108780)
If you find this code useful in your research, please consider citing:

    @article{RUIZBARROSO2025108780,
    title = {FADE: Forecasting for anomaly detection on ECG},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {267},
    pages = {108780},
    year = {2025},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2025.108780},
    author = {Paula Ruiz-Barroso and Francisco M. Castro and José Miranda and Denisa-Andreea Constantinescu and David Atienza and Nicolás Guil}}
