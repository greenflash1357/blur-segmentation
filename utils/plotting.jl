# helper functions for plotting

using PyPlot
using JLD
using HDF5
using Colors

include("images.jl") # imagae processing tools


function plot_segmentation(im, mask)
  
  gray = rgb2gray(im2double(im))
  hsv = zeros(size(im))
  hsv[:,:,1] = 60
  hsv[:,:,1] += (mask.>0.5)*180
  hsv[:,:,2] = 1
  hsv[:,:,3] = gray

  rgb = separate(convert(Image{RGB}, colorim(hsv, "HSV"))).data

  imshow(rgb, interpolation="none")
end

function plot_filters(path)
  filters = load(joinpath(path,"c1filters.jld"),"filters")
  figure()
  for i = 1:size(filters,4)
    #f = mean(filters[:,:,:,i],3)[:,:,1]
    f = permutedims(filters[:,:,:,i], [2,1,3])
    f = (f-minimum(f)) ./ (maximum(f)-minimum(f))
    subplot(8,12,i)
    imshow(f,interpolation="none")
    axis("off")
  end
  return gcf()
end

function plot_accuracy(path)
  stats = load(joinpath(path,"statistics.jld"))
  acc = stats["statistics"]["accuracy-accuracy"]
  i = sort(collect(keys(acc)))
  v = [acc[n] for n in i]
  figure()
  plot(i,v)
  grid("on")
  title("Test Accuracy during Training")
  xlabel("Iterations")
  ylabel("Accuracy")
  return gcf()
end

function plot_trainings_accuracy(path)
  stats = load(joinpath(path,"statistics.jld"))
  acc = stats["statistics"]["trainings-accuracy-accuracy"]
  i = sort(collect(keys(acc)))
  v = [acc[n] for n in i]
  figure()
  plot(i,v)
  grid("on")
  title("Trainings Accuracy during Training")
  xlabel("Iterations")
  ylabel("Accuracy")
  return gcf()
end

function plot_patches(filepath)
  file = h5open(filepath)
  data = read(file,"data")
  labels = read(file,"label")
  close(file)
  c = size(data,3)
  n = size(data,4)
  idxs = rand(1:n,25)
  figure()
  for i = 1:25
    subplot(5,5,i)
    if c == 3
      imshow(permutedims(data[:,:,:,idxs[i]],[2,1,3]),interpolation="none")
    elseif c == 2
      patch = sqrt(data[:,:,1,idxs[i]]'.^2 + data[:,:,2,idxs[i]]'.^2)
      imshow(patch,"gray",interpolation="none")
    elseif c == 1
      imshow(data[:,:,1,idxs[i]],"gray",interpolation="none")
    end
    if labels[idxs[i]] == 0
      title("sharp")
    else
      title("blurry")
    end
    axis("off")
  end
  suptitle("Random patch samples")
  return gcf()
end