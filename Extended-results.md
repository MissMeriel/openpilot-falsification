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

### Imageset 005

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 005-0](results-images/orig/imageset0005-0.jpg) |  ![Imageset 005-1](results-images/orig/imageset0005-1.jpg)


Output          | 
:-------------------------:|
![Imageset 005 plot](results-images/orig/imageset0005plot.jpg) |

### Imageset 102

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 102-0](results-images/orig/imageset0102-0.jpg) |  ![Imageset 102-1](results-images/orig/imageset0102-1.jpg)


Output          | 
:-------------------------:|
![Imageset 102 plot](results-images/orig/imageset0102plot.jpg) |

### Imageset 104

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 104-0](results-images/orig/imageset0104-0.jpg) |  ![Imageset 104-1](results-images/orig/imageset0104-1.jpg)


Output          | 
:-------------------------:|
![Imageset 104 plot](results-images/orig/imageset0104plot.jpg) |

### Imageset 199

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 199-0](results-images/orig/imageset0199-0.jpg) |  ![Imageset 199-1](results-images/orig/imageset0199-1.jpg)


Output          | 
:-------------------------:|
![Imageset 199 plot](results-images/orig/imageset0199plot.jpg) |

### Imageset 314

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 314-0](results-images/orig/imageset0314-0.jpg) |  ![Imageset 314-1](results-images/orig/imageset0314-1.jpg)


Output          | 
:-------------------------:|
![Imageset 314 plot](results-images/orig/imageset0314plot.jpg) |


### Imageset 390

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 390-0](results-images/orig/imageset0390-0.jpg) |  ![Imageset 390-1](results-images/orig/imageset0390-1.jpg)


Output          | 
:-------------------------:|
![Imageset 390 plot](results-images/orig/imageset0390plot.jpg) |

### Imageset 448

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 448-0](results-images/orig/imageset0448-0.jpg) |  ![Imageset 448-1](results-images/orig/imageset0448-1.jpg)


Output          | 
:-------------------------:|
![Imageset 448 plot](results-images/orig/imageset0448plot.jpg) |

### Imageset 475

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 475-0](results-images/orig/imageset0475-0.jpg) |  ![Imageset 475-1](results-images/orig/imageset0475-1.jpg)


Output          | 
:-------------------------:|
![Imageset 475 plot](results-images/orig/imageset0475plot.jpg) |

### Imageset 597

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 597-0](results-images/orig/imageset0597-0.jpg) |  ![Imageset 597-1](results-images/orig/imageset0597-1.jpg)


Output          | 
:-------------------------:|
![Imageset 597 plot](results-images/orig/imageset0597plot.jpg) |

### Imageset 680

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 680-0](results-images/orig/imageset0680-0.jpg) |  ![Imageset 680-1](results-images/orig/imageset0680-1.jpg)


Output          | 
:-------------------------:|
![Imageset 680 plot](results-images/orig/imageset0680plot.jpg) |

## Baseline Input Images

### Imageset 005

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 005-0 baseline](results-images/baseline/imageset0005-92-10.0-0.jpg) |  ![Imageset 005-1 baseline](results-images/baseline/imageset0005-92-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 005 baseline plot](results-images/baseline/imageset0005-92-10.0plot.jpg) |

### Imageset 102

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 102-0 baseline](results-images/baseline/imageset0102-135-10.0-0.jpg) |  ![Imageset 102-1 baseline](results-images/baseline/imageset0102-135-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 102 baseline plot](results-images/baseline/imageset0102-135-10.0plot.jpg) |

### Imageset 104

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 104-0 baseline](results-images/baseline/imageset0104-52-10.0-0.jpg) |  ![Imageset 104-1 baseline](results-images/baseline/imageset0104-52-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 104 baseline plot](results-images/baseline/imageset0104-52-10.0plot.jpg) |

### Imageset 199

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 199-0 baseline](results-images/baseline/imageset0199-237-10.0-0.jpg) |  ![Imageset 104-1 baseline](results-images/baseline/imageset0199-237-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 199 baseline plot](results-images/baseline/imageset0199-237-10.0plot.jpg) |

### Imageset 314

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 314-0 baseline](results-images/baseline/imageset0314-314-10.0-0.jpg) |  ![Imageset 314-1 baseline](results-images/baseline/imageset0314-314-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 314 baseline plot](results-images/baseline/imageset0314-314-10.0plot.jpg) |

### Imageset 390

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 390-0 baseline](results-images/baseline/imageset0390-406-10.0-0.jpg) |  ![Imageset 390-1 baseline](results-images/baseline/imageset0390-406-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 390 baseline plot](results-images/baseline/imageset0390-406-10.0plot.jpg) |

### Imageset 448

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 448-0 baseline](results-images/baseline/imageset0448-666-10.0-0.jpg) |  ![Imageset 448-1 baseline](results-images/baseline/imageset0448-666-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 448 baseline plot](results-images/baseline/imageset0448-666-10.0plot.jpg) |

### Imageset 475

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 475-0 baseline](results-images/baseline/imageset0475-158-10.0-0.jpg) |  ![Imageset 475-1 baseline](results-images/baseline/imageset0475-158-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 475 baseline plot](results-images/baseline/imageset0475-158-10.0plot.jpg) |

### Imageset 597
Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 597-0 baseline](results-images/baseline/imageset0597-743-10.0-0.jpg) |  ![Imageset 597-1 baseline](results-images/baseline/imageset0597-743-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 597 baseline plot](results-images/baseline/imageset0597-743-10.0plot.jpg) |

### Imageset 680

Image 0            |  Image 1
:-------------------------:|:-------------------------:
![Imageset 680-0 baseline](results-images/baseline/imageset0680-37-10.0-0.jpg) |  ![Imageset 680-1 baseline](results-images/baseline/imageset0680-37-10.0-1.jpg)


Output          | 
:-------------------------:|
![Imageset 680 baseline plot](results-images/baseline/imageset0680-37-10.0plot.jpg) |

## Counterexamples for Property 1

original imageset, original output, counterexample pair, counterexample output 

## Counterexamples for Property 2


## Counterexamples for Property 3

