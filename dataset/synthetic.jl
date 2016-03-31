using PyPlot
using Images
using Grid

include("../utils/images.jl") # image processing tools


function createmotionkernel(len, alpha)
	mid = div(len,2)+1
	a = zeros(len,len)
	a[mid,:] = [0.5 ones(1,len-2) 0.5]
	ai = InterpGrid(a, 0, InterpLinear)
	k = zeros(size(a))
	for r = 1:len
		for c = 1:len
			y = (r-mid)*cosd(alpha) + (c-mid)*sind(alpha) + mid
			x = (r-mid)*sind(alpha) + (c-mid)*cosd(alpha) + mid
			k[r,c] = ai[y,x]
		end
	end
	k /= sum(k)
	return k
end

function getrandomimages()
	segdir = "VOCdevkit/VOC2012/SegmentationClass"
	imagedir = "VOCdevkit/VOC2012/JPEGImages"
	n = length(readdir(segdir))
	labels = []
	fg = []
	bg = []
	seg = []
	# loop until randomly choosen cropped segmentation contains a valid class
	while length(labels) == 0
		# load images
		i = rand(1:n)
		segfile = readdir(segdir)[i]
		segpath = joinpath(segdir, segfile)
		fgfile = replace(segfile, ".png", ".jpg")
		fgpath = joinpath(imagedir, fgfile)
		bgpath = joinpath(imagedir, readdir(imagedir)[rand(1:length(readdir(imagedir)))])
		fg = im2double(PyPlot.imread(fgpath))
		bg = im2double(PyPlot.imread(bgpath))
		seg = PyPlot.imread(segpath)
		# crop images to common size
		w = min(size(fg,2), size(bg,2))
		h = min(size(fg,1), size(bg,1))
		fg = fg[1:h,1:w,:]
		bg = bg[1:h,1:w,:]
		seg = seg[1:h,1:w,:]
		# get mask from segmentation 
		seg=round(rgb2gray(seg)*255)
		labels = filter(x -> x!=0 && x!==220, unique(seg)) # remove undefined/ambigious/border classes
	end
	area = map(label->sum(seg.==label), labels) 
	idx = indmin(abs(area - length(seg)/2)) # choose class that consumes half of the image area
	mask = seg.==labels[idx]
	mask = float(mask)
	return fg, bg, mask
end

function createmotionblur(fg, bg, mask)
	k = createmotionkernel(rand(5:2:15), rand(0:180))
	blur = imfilter(fg .* mask, k) + (1 - imfilter(mask,k)) .* bg
	gt = float(imfilter(mask,k).>0)
	return blur, gt
end

function createfocalblur(fg, bg, mask)
	k = gaussian2d(rand(1:0.1:2.5),[9 9])
	bg_blurred = imfilter(bg, k)
	s = gaussian2d(0.5,[5 5])
	blur = fg.*imfilter(mask, s) + (1 - imfilter(mask,s)) .* bg_blurred
	gt = 1-mask
	return blur, gt
end

# Create synthetic dataset

for folder in ["synthetic/image/motion", "synthetic/image/focus", 
			   "synthetic/gt/motion", "synthetic/gt/focus"]
	if !ispath(folder)
		mkpath(folder)
	end
end

tic()
for i = 1:1000
	println("Getting motion nr. $(i)...")
	fg, bg, mask = getrandomimages()
	motion, gt = createmotionblur(fg, bg, mask)
	save(joinpath("synthetic/image/motion", @sprintf("motion%04i.jpg",i)), convert(Image, motion))
	save(joinpath("synthetic/gt/motion", @sprintf("motion%04i.png",i)), grayim(gt')) # Due to an issue in Images.jl gt gets transposed!

	println("Getting out_of_focus nr. $(i)...")
	fg, bg, mask = getrandomimages()
	focus, gt = createfocalblur(fg, bg, mask)
	save(joinpath("synthetic/image/focus", @sprintf("out_of_focus%04i.jpg",i)), convert(Image, focus))
	save(joinpath("synthetic/gt/focus", @sprintf("out_of_focus%04i.png",i)), grayim(gt')) # Due to an issue in Images.jl gt gets transposed!
end
toc()