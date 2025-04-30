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


### Our [paper](https://doi.org/10.1016/j.cmpb.2025.108780)
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
