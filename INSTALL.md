## Get started
1. Clone the MFIALane repository
    ```
    git clone https://github.com/Cuibaby/MFIALane.git
    ```
    We call this directory as `$MFIALane_ROOT`

2. Create a conda virtual environment and activate it (conda is optional)

    ```Shell
    conda create -n MFIALane python=3.8 -y
    conda activate MFIALane
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

    # Or you can install via pip
    pip install torch torchvision

    # Install python packages
    pip install -r requirements.txt
    ```

4. Data preparation

    Download [VIL100](https://github.com/yujun0-0/MMA-Net/tree/main/dataset), [CULane](https://xingangpan.github.io/projects/CULane.html) and [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to  `$VIL100ROOT` `$CULANEROOT` and `$TUSIMPLEROOT`. Create link to `data` directory.
    
    ```Shell
    cd $MFIALane_ROOT
    mkdir -p data
    ln -s $VIL-100ROOT data/VIL100
    ln -s $CULANEROOT data/CULane
    ln -s $TUSIMPLEROOT data/tusimple
    ```

    For CULane, you should have structure like this:
    ```
    $CULANEROOT/driver_xx_xxframe    # data folders x6
    $CULANEROOT/laneseg_label_w16    # lane segmentation labels
    $CULANEROOT/list                 # data lists
    ```

    For Tusimple, you should have structure like this:
    ```
    $TUSIMPLEROOT/clips # data folders
    $TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
    $TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
    $TUSIMPLEROOT/test_label.json # test label json file

    ```

    For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

    ```Shell
    python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
    # this will generate seg_label directory
    ```
    For VIL100, you should have structure like this:
    ```
    $VIL100ROOT/Annotations 
    $VIL100ROOT/data 
    $VIL100ROOT/JPEGImages
    $VIL100ROOT/Json
    $VIL100ROOT/list
    $VIL100ROOT/test

    ```

5. Install CULane evaluation tools. 

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++.  Or just install opencv with command `sudo apt-get install libopencv-dev`

    
    Then compile the evaluation tool of CULane.
    ```Shell
    cd $MFIALane_ROOT/runner/evaluator/culane/lane_evaluation
    make
    cd -
    ```
    
    Note that, the default `opencv` version is 3. If you use opencv2, please modify the `OPENCV_VERSION := 3` to `OPENCV_VERSION := 2` in the `Makefile`.
    
    If you have problems installing the C++ version, you can remove the $lane_evaluation and change the 'type=Py_CULane' in the config file to use the pure Python version for evaluation.

## Training

For training, run

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py [configs/path_to_your_config] --gpus [gpu_ids]
```


For example, run
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py configs/culane.py --gpus 0 1 2 3
```

## Testing
For testing, run
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py c[configs/path_to_your_config] --validate --load_from [path_to_your_model] [gpu_num]
```

For example, run
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py configs/culane.py --validate --load_from culane.pth --gpus 0 1 2 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py configs/tusimple.py --validate --load_from tusimple.pth --gpus 0 1 2 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py configs/vilane.py --validate --load_from vilane.pth --gpus 0 1 2 3
```
