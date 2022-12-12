import torch

datalocation = 'E:\\CJC\\SVR\\project\\SVR-pytorch3d1- 系列\\SVSR-Tank'
device = torch.device("cuda:0")
objRoot = 'E:\\CJC\\SVR\\project\\data\\new\\objdata-simple2\\'
objTrainList = 'data/train_list.txt'

saveroot = './results'

faces_per_pixel = 5
workers = 0
nEpoch = 1000
plot_period = 10
save_period = 100
batchsize = 1
partindex_n = 0

num_views = 20
sigma_re = 7e-6
num_views_per_iteration = 3
shuffle_ = True
vertexscale = 0.98
imagesize = 256

randomviewlist = [0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
losses = {
          # "rgb": {"weight": 0.0001, "values": []},
          "silhouette": {"weight": 1, "values": []},
          "edge": {"weight": 1, "values": []},
          "normal": {"weight": 0.05, "values": []},
          "laplacian": {"weight": 0.8, "values": []},
          'chamfer': {"weight": 0.0, "values": []},
         }