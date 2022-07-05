# Extended results from the paper

|                                            | Falsification Rate | Avg. Total  Time (s) | Avg. Falsification  Time (s) | Average  Restarts | Images with Counterexamples |
|--------------------------------------------|:------------------:|:--------------------:|:----------------------------:|:-----------------:|:---------------------------:|
| Property 1: nearest lanes confidence       |                17% |                222.6 |                        142.8 |              88.2 |                           4 |
| Property 2: last y-values of nearest lanes |                47% |                123.6 |                         44.6 |              60.3 |                           6 |
| Property 3: lead car confidence            |                58% |                197.4 |                         46.5 |              46.5 |                           6 |

The above results show average falsification performance for ten images within _epsilon_=10.
The distance metric determining epsilon was computed using _L-infinity_, meaning that the distance was calculated according to the maximum difference of any value in the generated input from the original input. 

## Original Input Images

Original images were chosen for the variety of the environment shown in the input imageset and the interpretability and stability of the network output.
Each imageset is comprised of two consecutive images, Image 0 and Image 1, collected by an onboard camera during "normal" driving on highway and surface roads. 
These two images are reordered into a composite tensor to be fed to the supercombo network as one input.
The original imagesets are shown with their corresponding output from the network.
The other 3 inputs were given default values. Refer to [property definitions](falsification/properties) for default values.
To download this dataset for yourself, refer to the [dataset README](dataset/README.md).

### Imageset 005

Image 0            |  Image 1        | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 005-0](results-images/orig/imageset0005-0.jpg) |  ![Imageset 005-1](results-images/orig/imageset0005-1.jpg) | ![Imageset 005 plot](results-images/orig/imageset0005plot.jpg) 


### Imageset 102

Image 0            |  Image 1      | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 102-0](results-images/orig/imageset0102-0.jpg) |  ![Imageset 102-1](results-images/orig/imageset0102-1.jpg) | ![Imageset 102 plot](results-images/orig/imageset0102plot.jpg) 


### Imageset 104

Image 0            |  Image 1     | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 104-0](results-images/orig/imageset0104-0.jpg) |  ![Imageset 104-1](results-images/orig/imageset0104-1.jpg) | ![Imageset 104 plot](results-images/orig/imageset0104plot.jpg) 


### Imageset 199

Image 0            |  Image 1     | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 199-0](results-images/orig/imageset0199-0.jpg) |  ![Imageset 199-1](results-images/orig/imageset0199-1.jpg) | ![Imageset 199 plot](results-images/orig/imageset0199plot.jpg) 


### Imageset 314

Image 0            |  Image 1     | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 314-0](results-images/orig/imageset0314-0.jpg) |  ![Imageset 314-1](results-images/orig/imageset0314-1.jpg)  | ![Imageset 314 plot](results-images/orig/imageset0314plot.jpg)


### Imageset 390

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 390-0](results-images/orig/imageset0390-0.jpg) |  ![Imageset 390-1](results-images/orig/imageset0390-1.jpg)   | ![Imageset 390 plot](results-images/orig/imageset0390plot.jpg)


### Imageset 448

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 448-0](results-images/orig/imageset0448-0.jpg) |  ![Imageset 448-1](results-images/orig/imageset0448-1.jpg)  | ![Imageset 448 plot](results-images/orig/imageset0448plot.jpg)


### Imageset 475

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 475-0](results-images/orig/imageset0475-0.jpg) |  ![Imageset 475-1](results-images/orig/imageset0475-1.jpg)  | ![Imageset 475 plot](results-images/orig/imageset0475plot.jpg)


### Imageset 597

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 597-0](results-images/orig/imageset0597-0.jpg) |  ![Imageset 597-1](results-images/orig/imageset0597-1.jpg)  | ![Imageset 597 plot](results-images/orig/imageset0597plot.jpg)


### Imageset 680

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 680-0](results-images/orig/imageset0680-0.jpg) |  ![Imageset 680-1](results-images/orig/imageset0680-1.jpg) | ![Imageset 680 plot](results-images/orig/imageset0680plot.jpg)


## Baseline Input Images

The baseline images were generated by sampling noise from a Gaussian distribution and applying them to the original input images.
Baseline images with an L-infinity distance of 10 from the original images were included in this study.
Just like the original images, the baseline Image 0 and Image 1 are preprocessed into a single composite tensor before being passed to the network.
The purpose of the baseline is to determine the susceptibility of the network to untargeted changes in the image inputs.
As displayed in the baseline images below, the change in output compared to the original image inputs is minimal.
The changes in output do not violate the limits given in the [property definitions](falsification/properties).

### Imageset 005

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 005-0 baseline](results-images/baseline/imageset0005-92-10.0-0.jpg) |  ![Imageset 005-1 baseline](results-images/baseline/imageset0005-92-10.0-1.jpg)  | ![Imageset 005 baseline plot](results-images/baseline/imageset0005-92-10.0plot.jpg) 


### Imageset 102

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 102-0 baseline](results-images/baseline/imageset0102-135-10.0-0.jpg) |  ![Imageset 102-1 baseline](results-images/baseline/imageset0102-135-10.0-1.jpg)  | ![Imageset 102 baseline plot](results-images/baseline/imageset0102-135-10.0plot.jpg)


### Imageset 104

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 104-0 baseline](results-images/baseline/imageset0104-52-10.0-0.jpg) |  ![Imageset 104-1 baseline](results-images/baseline/imageset0104-52-10.0-1.jpg) | ![Imageset 104 baseline plot](results-images/baseline/imageset0104-52-10.0plot.jpg)



### Imageset 199

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 199-0 baseline](results-images/baseline/imageset0199-237-10.0-0.jpg) |  ![Imageset 104-1 baseline](results-images/baseline/imageset0199-237-10.0-1.jpg)  | ![Imageset 199 baseline plot](results-images/baseline/imageset0199-237-10.0plot.jpg)


### Imageset 314

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 314-0 baseline](results-images/baseline/imageset0314-314-10.0-0.jpg) |  ![Imageset 314-1 baseline](results-images/baseline/imageset0314-314-10.0-1.jpg) | ![Imageset 314 baseline plot](results-images/baseline/imageset0314-314-10.0plot.jpg)


### Imageset 390

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 390-0 baseline](results-images/baseline/imageset0390-406-10.0-0.jpg) |  ![Imageset 390-1 baseline](results-images/baseline/imageset0390-406-10.0-1.jpg) | ![Imageset 390 baseline plot](results-images/baseline/imageset0390-406-10.0plot.jpg)


### Imageset 448

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 448-0 baseline](results-images/baseline/imageset0448-666-10.0-0.jpg) |  ![Imageset 448-1 baseline](results-images/baseline/imageset0448-666-10.0-1.jpg) | ![Imageset 448 baseline plot](results-images/baseline/imageset0448-666-10.0plot.jpg) 


### Imageset 475

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 475-0 baseline](results-images/baseline/imageset0475-158-10.0-0.jpg) |  ![Imageset 475-1 baseline](results-images/baseline/imageset0475-158-10.0-1.jpg) | ![Imageset 475 baseline plot](results-images/baseline/imageset0475-158-10.0plot.jpg)


### Imageset 597
Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 597-0 baseline](results-images/baseline/imageset0597-743-10.0-0.jpg) |  ![Imageset 597-1 baseline](results-images/baseline/imageset0597-743-10.0-1.jpg) | ![Imageset 597 baseline plot](results-images/baseline/imageset0597-743-10.0plot.jpg) 


### Imageset 680

Image 0            |  Image 1          | Output
:-------------------------:|:-------------------------:|:-------------------------:
![Imageset 680-0 baseline](results-images/baseline/imageset0680-37-10.0-0.jpg) |  ![Imageset 680-1 baseline](results-images/baseline/imageset0680-37-10.0-1.jpg) | ![Imageset 680 baseline plot](results-images/baseline/imageset0680-37-10.0plot.jpg) 


## Counterexamples for Property 1


|                | Imageset 005                             | Imageset 102 | Imageset 104 | Imageset 199 | Imageset 314 |
|----------------|------------------------------------------|--------------|--------------|--------------|--------------|
| Original Image 0 | ![Imageset 005-0](results-images/orig/imageset0005-0.jpg) | ![Imageset 102-0](results-images/orig/imageset0102-0.jpg) | ![Imageset 104-0](results-images/orig/imageset0104-0.jpg) |  ![Imageset 199-0](results-images/orig/imageset0199-0.jpg) | ![Imageset 314-0](results-images/orig/imageset0314-0.jpg) 
| Counterexample Image 0 | ![](results-images/counterexamples/counterexample_imageset0005_prop1_10_test_eps10-perc10-0.jpg) | ![](results-images/counterexamples/counterexample_imageset0102_prop1_10_test_eps10-perc10-0.jpg) | ![](results-images/counterexamples/counterexample_imageset0104_prop1_1_test_eps10-perc10-0.jpg) | N/A          | N/A          |
| Output         | ![](results-images/counterexamples/counterexample_imageset0005_prop1_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0102_prop1_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0104_prop1_1_test_eps10-perc10plot.jpg) | N/A          | N/A          |

|                | Imageset 390 | Imageset 448 | Imageset 475                                                                                      | Imageset 597                                                                                      | Imageset 680 |
|----------------|--------------|--------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------|
| Original Image 0 | ![Imageset 390-0](results-images/orig/imageset0390-0.jpg) | ![Imageset 448-0](results-images/orig/imageset0448-0.jpg) | ![Imageset 475-0](results-images/orig/imageset0475-0.jpg) |  ![Imageset 597-0](results-images/orig/imageset0597-0.jpg) | ![Imageset 680-0](results-images/orig/imageset0680-0.jpg) 
| Counterexample Image 0 | N/A | N/A | ![](results-images/counterexamples/counterexample_imageset0475_prop1_9_test_eps10-perc10-0.jpg)   | ![](results-images/counterexamples/counterexample_imageset0597_prop1_2_test_eps10-perc10-0.jpg)   | N/A |
| Output         | N/A | N/A | ![](results-images/counterexamples/counterexample_imageset0475_prop1_9_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0597_prop1_2_test_eps10-perc10plot.jpg) | N/A |

## Counterexamples for Property 2


|                | Imageset 005 | Imageset 102 | Imageset 104                                                                                       | Imageset 199                                                                                       | Imageset 314 |
|----------------|--------------|--------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------|
| Original Image 0 | ![Imageset 005-0](results-images/orig/imageset0005-0.jpg) | ![Imageset 102-0](results-images/orig/imageset0102-0.jpg) | ![Imageset 104-0](results-images/orig/imageset0104-0.jpg) |  ![Imageset 199-0](results-images/orig/imageset0199-0.jpg) | ![Imageset 314-0](results-images/orig/imageset0314-0.jpg) 
| Counterexample Image 0 | N/A          | ![](results-images/counterexamples/counterexample_imageset0102_prop2_10_test_eps10-perc10-0.jpg) | ![](results-images/counterexamples/counterexample_imageset0104_prop2_10_test_eps10-perc10-0.jpg)   | ![](results-images/counterexamples/counterexample_imageset0199_prop2_10_test_eps10-perc10-0.jpg)   | N/A |
| Output         | N/A          | ![](results-images/counterexamples/counterexample_imageset0102_prop2_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0104_prop2_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0199_prop2_10_test_eps10-perc10plot.jpg) | N/A |

|                | Imageset 390                             | Imageset 448                                                                                       | Imageset 475                                                                                      | Imageset 597 | Imageset 680                                                                                      |
|----------------|------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------|
| Original Image 0 | ![Imageset 390-0](results-images/orig/imageset0390-0.jpg) | ![Imageset 448-0](results-images/orig/imageset0448-0.jpg) | ![Imageset 475-0](results-images/orig/imageset0475-0.jpg) |  ![Imageset 597-0](results-images/orig/imageset0597-0.jpg) | ![Imageset 680-0](results-images/orig/imageset0680-0.jpg) 
| Counterexample Image 0 | N/A                                      | ![](results-images/counterexamples/counterexample_imageset0448_prop2_10_test_eps10-perc10-0.jpg)   | ![](results-images/counterexamples/counterexample_imageset0475_prop2_2_test_eps10-perc10-0.jpg)   | N/A | ![](results-images/counterexamples/counterexample_imageset0680_prop2_1_test_eps10-perc10-0.jpg)   |
| Output         | N/A | ![](results-images/counterexamples/counterexample_imageset0448_prop2_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0475_prop2_2_test_eps10-perc10plot.jpg) | N/A | ![](results-images/counterexamples/counterexample_imageset0680_prop2_1_test_eps10-perc10plot.jpg) |

## Counterexamples for Property 3


|                | Imageset 005                             | Imageset 102                                                                                        | Imageset 104                                                                                       | Imageset 199                                                                                       | Imageset 314                                                                                       |
|----------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Original Image 0 | ![Imageset 005-0](results-images/orig/imageset0005-0.jpg) | ![Imageset 102-0](results-images/orig/imageset0102-0.jpg) | ![Imageset 104-0](results-images/orig/imageset0104-0.jpg) |  ![Imageset 199-0](results-images/orig/imageset0199-0.jpg) | ![Imageset 314-0](results-images/orig/imageset0314-0.jpg) 
| Counterexample Image 0 | ![](results-images/counterexamples/counterexample_imageset0005_prop3_2_test_eps10-perc10-0.jpg) | ![](results-images/counterexamples/counterexample_imageset0102_prop3_10_test_eps10-perc10-0.jpg)    | ![](results-images/counterexamples/counterexample_imageset0104_prop3_10_test_eps10-perc10-0.jpg)   | ![](results-images/counterexamples/counterexample_imageset0199_prop3_10_test_eps10-perc10-0.jpg)   | ![](results-images/counterexamples/counterexample_imageset0314_prop3_10_test_eps10-perc10-0.jpg)   |
| Output         | ![](results-images/counterexamples/counterexample_imageset0005_prop3_2_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0102_prop3_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0104_prop3_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0199_prop3_10_test_eps10-perc10plot.jpg) | ![](results-images/counterexamples/counterexample_imageset0314_prop3_10_test_eps10-perc10plot.jpg) |

|                | Imageset 390 | Imageset 448 | Imageset 475 | Imageset 597                                                                                       | Imageset 680 |
|----------------|--------------|--------------|--------------|----------------------------------------------------------------------------------------------------|--------------|
| Original Image 0 | ![Imageset 390-0](results-images/orig/imageset0390-0.jpg) | ![Imageset 448-0](results-images/orig/imageset0448-0.jpg) | ![Imageset 475-0](results-images/orig/imageset0475-0.jpg) |  ![Imageset 597-0](results-images/orig/imageset0597-0.jpg) | ![Imageset 680-0](results-images/orig/imageset0680-0.jpg) 
| Counterexample Image 0| N/A | N/A | N/A | ![](results-images/counterexamples/counterexample_imageset0597_prop3_10_test_eps10-perc10-0.jpg)   | N/A |
| Output         | N/A | N/A | N/A | ![](results-images/counterexamples/counterexample_imageset0597_prop3_10_test_eps10-perc10plot.jpg) | N/A |
