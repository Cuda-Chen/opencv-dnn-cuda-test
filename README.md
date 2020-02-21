# opencv-dnn-cuda-test
Sample code of testing functions of OpenCV with CUDA-enabled DNN modules.

# Setting up
```
$ cd yolos
$ wget https://pjreddie.com/media/files/yolov3.weights
$ cd ..
```

Then go to either [Python](https://github.com/Cuda-Chen/opencv-dnn-cuda-test/tree/master/python_code)
 or [C++](https://github.com/Cuda-Chen/opencv-dnn-cuda-test/tree/master/cpp_code) part to validate the installation of OpenCV
with CUDA-enabled DNN modules.

# Special Thanks
Thanks for [YashasSamaga](https://github.com/YashasSamaga) providing a quick dirty way to measure FPS.
You can visit his work about this:
- https://github.com/opencv/opencv/pull/14827#issuecomment-574086174

As C++ part, I adapted some lines from the following (written by YashasSamaga):
- https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

Also, he provides the way to compile OpenCV version 4.x without errors:
- https://github.com/opencv/opencv/issues/16439#issuecomment-578521193
