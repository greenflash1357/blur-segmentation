## Utils

### Image processing

* `minfilter`: Provides a minimum filter for grayscale images, using a squared kernel of a given size (always odd).
* `rgb2gray`: Performs gray scale conversion for color images, using [Rec. 601 Luma](https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems).
* `im2double`: Divides integer images by their `typemax`. This is most commonly used to scale `UInt8` images into unit range, by dividing them by 255.

### Plotting

* `plot_segmentation`: Shows a colored overlay of a binary mask onto an image.
* `plot_filters`: Shows the 96 filter kernels of conv1 layer.
* `plot_accuracy`: Shows the patch based accuarcy agains the trainings iterations.
* `plot_patches`: Shows a random sample of the extracted patches.

### I/O

* `save_figure`: Saves a PyPlot figure object as image.
* `extract_filters`: Stores the parameters of conv1 layer as `c1filters.jld`.