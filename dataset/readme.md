## Data

### Getting raw data

As basis for training and evaluation we use the [Blur Detection Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/dataset.html). Or artificially blurred images based on images from the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

You can run `build.jl` to download, extract and split the datasets. Or do it manually. The desired folder structure is:
```
/dataset
|--/blurdetect
|  |--/gt
|  |  |--/motion
|  |  |--/focus
|  |--/image
|     |--/motion
|     |--/focus
|--/evalset
|  |--/gt
|  |--/image
|--/VOCdevkit
   |--/VOC2012
   |--/JPEGImages
   |--/SegmentationClass
   |--/SegmentationObject
```

### Artificial blur

To synthesize artificially blurred images, run `synthetic.jl`. This will generate 1000 images for each blur type, motion and defocus.

### Extracting patches

Run `patches.jl` to generate the final image patches for training. This will store a training and testing set of patches as `HDF5` files. Note: For the current setting, the patch sets will consume over 10 GB of disk space.

You can adjust the number of patches, the ratio of the blur types or the class labels inside the script. There is also a switch to select the dataset.