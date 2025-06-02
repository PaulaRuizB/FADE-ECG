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

     _Note: To include the signal quality index for the NSR dataset, please download the [ECG_QC library](https://github.com/Aura-healthcare/ecg_qc)_
     
   * MIT-BIH Arrhythmia database: run ```datasets/MIT-Arr_rhythms_beats.py``` after specifying the ```path``` (location of the raw dataset) and ```save_path``` (destination for the processed data) arguments.
   
### Train baseline forecasting model with PyTorch:
```
python -m torch.distributed.run --nproc_per_node=4 ../mains/train_original.py --prefix exp --loss2 None --optim_name adamw --use_scheduler --warmup 0 --path_datasetinfo /path/dataset_info1.h5 --path_train_set /path/train_set1/ --save_every 25 --num_channels 1 --loss1 mse_inside_outside_thresholds --add_FC --batch_size 256 --sqi 0.5 --dropout 0.2 --lr 1e-4 --min_lr 1e-7 --total_epochs 200 --weights_mse_inorout_lastsecond 5.0 1.0 --threshold_mse_inside_outside 0.4 -0.4 --norm_a_b_max_min --save_path_train /path/experiments/ --input_size_seconds 4
```

### Train domain-adapted model with PyTorch:
```
python -m torch.distributed.run --nproc_per_node=4 ../mains/train_original.py --prefix repo --loss2 None --optim_name adamw --use_scheduler --warmup 0 --path_datasetinfo /path/dataset_info2.h5 --path_train_set /path/train_set2/ --path_test_set /path_test_set/ --save_every 25 --num_channels 1 --loss1 mse_inside_outside_thresholds --add_FC --batch_size 64 --sqi -1 --dropout 0.2 --lr 1e-4 --min_lr 1e-8 --total_epochs 250 --weights_mse_inorout_lastsecond 1.0 1.0 --threshold_mse_inside_outside 0.3 -0.3 --ft --dataset_FT mit_arr_beats --input_size_seconds 4
```

### Test forecasting models with PyTorch:
```
python ../mains/test_threshold_and_acc.py --path_datasetinfo /path/dataset_info2.h5 --path_test_set /path_test_set/ --path_trained_model /path_pretrained_model/ --seed 23 --input_size_seconds 4
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
