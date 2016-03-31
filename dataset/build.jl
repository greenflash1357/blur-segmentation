using BinDeps


# get blur detection dataset
println("Downloading images...")
run(download_cmd("http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetImage.zip",
				 "images.zip"))
println("Downloading ground truth...")
run(download_cmd("http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/data/BlurDatasetGT.zip",
				"gt.zip"))
println("Unpacking files...")
run(unpack_cmd("images.zip", "blurdetect", ".zip", ""))
run(unpack_cmd("gt.zip", "blurdetect", ".zip", ""))
rm("images.zip")
rm("gt.zip")

println("Moving files...")
# split motion and focus images
for dir = ["blurdetect/image","blurdetect/gt"]
	motionfiles = filter(x->(startswith(x,"motion") && isfile(x)), readdir(dir))
	focusfiles = filter(x->(startswith(x,"out_of_focus") && isfile(x)), readdir(dir))
	mkpath(joinpath(dir,"motion"))
	mkpath(joinpath(dir,"focus"))
	for file in motionfiles
		mv(joinpath(dir,file), joinpath(dir,"motion",file))
	end
	for file in focusfiles
		mv(joinpath(dir,file), joinpath(dir,"focus",file))
	end
end

println("Sampling evaluation images...")
# extract evaluation images
mkpath("evalset/image")
mkpath("evalset/gt")
for dir in ["motion", "focus"]
	for idx in randperm(length(readdir(joinpath("blurdetect/image",dir))))[1:10]
		srcdir = joinpath("blurdetect/image",dir)
		srcfiles = readdir(srcdir)
		mv(joinpath(srcdir, srcfiles[idx]), joinpath("evalset/image",srcfiles[idx]))
		srcdir = joinpath("blurdetect/gt",dir)
		srcfiles = readdir(srcdir)
		mv(joinpath(srcdir, srcfiles[idx]), joinpath("evalset/gt",srcfiles[idx]))
	end
end

println("Downloading Pascal VOC...")
# get Pascal VOC
run(download_cmd("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
				 "pascal.tar"))
println("Unpacking Pascal VOC...")
@unix_only cmd = unpack_cmd("pascal.tar", "", ".tar", "")
@windows_only cmd = `7z x pascal.tar -y`
run(cmd)
rm("pascal.tar")