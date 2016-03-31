# helper functions for i/o tasks

using JLD
using HDF5


function extract_filters(path)
  files = readdir(path)
  lastsnap = filter(x -> startswith(x,"snapshot"),files)[end]
  parameter = load(joinpath(path,lastsnap),"params_all")
  filters = parameter["conv1"][1]
  res = joinpath(path,"c1filters.jld")
  save(res,"filters",filters)
  return res
end

function save_figure(path)
  fig = gcf()
  file = open(path,"w")
  writemime(file,"image/png",fig)
  close(file)
  @assert isfile(path)
  return path
end
