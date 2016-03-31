import JLD
import HDF5
import PyPlot
import Images

include("../utils/images.jl") # image processing tools

# Return extracted patches and labels from given files.
# n is the number of patches per image.
# the file lists contain full paths
function getpatches(impathlist, gtpathlist, n)
  
  l = length(impathlist)
  patches = Array(Float32,31,31,3,l*n)
  labels = Array(Float32,l*n)
  i = 0

  for (impath,gtpath) in zip(impathlist,gtpathlist) # loop over motion images
    im = PyPlot.imread(impath)
    im = im2double(im)
    gt = PyPlot.imread(gtpath)
    gt = gt[:,:,1,1]
    r,c = size(im,1,2)
    reg1 = minfilter(gt, 31)
    reg2 = minfilter(1-gt, 31)
    reg3 = (1 - reg1) - reg2
    idxs1 = shuffle!(find(reg1))[1:min(div(n,3),length(find(reg1)))]
    idxs2 = shuffle!(find(reg2))[1:min(div(n,3),length(find(reg2)))]
    idxs3 = shuffle!(find(reg3))[1:min(div(n,3),length(find(reg3)))]
    idxs = [idxs1; idxs2; idxs3]
    #idxs = randperm(r*c)[1:n] # randomly selected pixels (linear index)
    ys,xs =ind2sub((r,c),idxs)
    padim = Images.padarray(im,[15,15,0],[15,15,0],"reflect")
    for (y,x) in zip(ys,xs) # loop over selected pixels
      i += 1
      labels[i] = gt[y,x]
      patch31 = padim[y:y+30,x:x+30,:]
      patches[:,:,:,i] = permutedims(patch31,[2,1,3])
    end
  end
  patches = patches[:,:,:,1:i]
  labels = labels[1:i]

  return patches, labels
end

tic()
println("Starting patch extraction...")

# Options
N = 1300000 # approximate total count of all patches (motion + focus, training + testing)
multiclass = false # if true, motion and focal blur have different class labels (sharp: 0, motion: 1, defocus: 2)
ratio = 0.5 # ratio of motion and focus patches (1.0: only motion patches, 0.0: only focus patches)
#datadir = "blurdetect" # blur detection database
datadir = "synthetic" # artificial blur database

motionimagedir = joinpath(datadir,"image","motion")
motiongtdir = joinpath(datadir,"gt","motion")
focusimagedir = joinpath(datadir,"image","focus")
focusgtdir = joinpath(datadir,"gt","focus")

motionimagefiles = readdir(motionimagedir)
motiongtfiles = readdir(motiongtdir)
focusimagefiles = readdir(focusimagedir)
focusgtfiles = readdir(focusgtdir)

nmotion = length(motionimagefiles)
nfocus = length(focusimagefiles)

nm = Int(div(N*ratio,nmotion)) # number of motion patches per image
nf = Int(div(N*(1.0-ratio),nfocus)) # number of focal patches per image

# split images into training and testing images (4:1)
motionidxs = randperm(nmotion)
focusidxs = randperm(nfocus)
trainmotionidxs = motionidxs[1:round(Int,nmotion*0.8)]
testmotionidxs = motionidxs[round(Int,nmotion*0.8)+1:end]
trainfocusidxs = focusidxs[1:round(Int,nfocus*0.8)]
testfocusidxs = focusidxs[round(Int,nfocus*0.8)+1:end]

trainmotionimagepaths = map(x->joinpath(motionimagedir, x), motionimagefiles[trainmotionidxs])
trainmotiongtpaths = map(x->joinpath(motiongtdir, x), motiongtfiles[trainmotionidxs])
testmotionimagepaths = map(x->joinpath(motionimagedir, x), motionimagefiles[testmotionidxs])
testmotiongtpaths = map(x->joinpath(motiongtdir, x), motiongtfiles[testmotionidxs])
trainfocusimagepaths = map(x->joinpath(focusimagedir, x), focusimagefiles[trainfocusidxs])
trainfocusgtpaths = map(x->joinpath(focusgtdir, x), focusgtfiles[trainfocusidxs])
testfocusimagepaths = map(x->joinpath(focusimagedir, x), focusimagefiles[testfocusidxs])
testfocusgtpaths = map(x->joinpath(focusgtdir, x), focusgtfiles[testfocusidxs])

# get patches
println("Getting training patches...")
trainmotionpatches, trainmotionlabels = getpatches(trainmotionimagepaths, trainmotiongtpaths, nm)
trainfocuspatches, trainfocuslabels = getpatches(trainfocusimagepaths, trainfocusgtpaths, nf)
println("Getting testing patches...")
testmotionpatches, testmotionlabels = getpatches(testmotionimagepaths, testmotiongtpaths, nm)
testfocuspatches, testfocuslabels = getpatches(testfocusimagepaths, testfocusgtpaths, nf)

if multiclass
  trainfocuslabels *= 2
  testfocuslabels *= 2
end

testpatches = cat(4, testmotionpatches, testfocuspatches)
testlabels = [testmotionlabels; testfocuslabels]
trainpatches = cat(4, trainmotionpatches, trainfocuspatches)
trainlabels = [trainmotionlabels; trainfocuslabels]

# shuffle training patches
randtrainidxs = randperm(length(trainlabels))
trainpatches = trainpatches[:,:,:,randtrainidxs]
trainlabels = trainlabels[randtrainidxs]

ntotal = length(trainlabels) + length(testlabels)
println("$(ntotal) patches extracted.")

# save as hdf5 files
HDF5.h5open("train.hdf5","w") do file
  HDF5.write(file,"data",trainpatches)
  HDF5.write(file,"label",trainlabels)
end
HDF5.h5open("test.hdf5","w") do file
  HDF5.write(file,"data",testpatches)
  HDF5.write(file,"label",testlabels)
end

println("All patches saved.")
toc()
