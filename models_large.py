import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import *

import os
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj



class Encoder(nn.Module):
    def __init__(self, dim_in=3, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1 * 2, dim1 * 4, int(dim2 / 4), dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=4, stride=2, padding=1)  # 128
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=4, stride=2, padding=1)  # 64
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=4, stride=2, padding=1)  # 32
        self.conv4 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=4, stride=2, padding=1)  # 16
        self.conv5 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=4, stride=2, padding=1)  # 8

        self.conv2_2 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=4, stride=4, padding=0)  # 16
        self.conv3_2 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 16
        self.conv4_2 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8

        self.conv2_3 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8
        self.conv3_3 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8

        self.conv4_2 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8

        self.conv2_3 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8
        self.conv3_3 = nn.Conv2d(dim_hidden[2], dim_hidden[2], kernel_size=3, stride=2, padding=1)  # 8
        # self.conv4_2 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1)  # 8
        # self.conv2_3 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1)  # 8
        # self.conv3_3 = nn.MaxPool2d( kernel_size=3, stride=2, padding=1)  # 8

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.bn4 = nn.BatchNorm2d(dim_hidden[2])
        self.bn5 = nn.BatchNorm2d(dim_hidden[2])
        self.bn2_2 = nn.BatchNorm2d(dim_hidden[2])
        self.bn3_2 = nn.BatchNorm2d(dim_hidden[2])
        self.bn4_2 = nn.BatchNorm2d(dim_hidden[2])
        self.bn2_3 = nn.BatchNorm2d(dim_hidden[2])
        self.bn3_3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1_0 = nn.Linear(dim_hidden[2] * 8 ** 2, dim_hidden[3])
        self.fc1_1 = nn.Linear(dim_hidden[2] * 8 ** 2, dim_hidden[3])
        self.fc1_2 = nn.Linear(dim_hidden[2] * 8 ** 2, dim_hidden[3])
        self.fc1_3 = nn.Linear(dim_hidden[2] * 8 ** 2, dim_hidden[3])

        self.fc2 = nn.Linear(dim_hidden[3] * 4, dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out - 50)
        self.fc3_6 = nn.Linear(dim_out - 50, 15)
        #
        self.fc6 = nn.Linear(15, 60)
        self.fc7 = nn.Linear(60, 50)

    def forward(self, x,view = 0):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.2, inplace=True)
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.2, inplace=True)
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.2, inplace=True)

        x2 = F.leaky_relu(self.bn2_2(self.conv2_2(x2)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.bn2_3(self.conv2_3(x2)), 0.2, inplace=True)
        # x2 = self.conv2_3(x2)
        x2 = x2.contiguous().view(x2.size(0), -1)
        x2 = F.relu(self.fc1_0(x2), inplace=True)

        x3 = F.leaky_relu(self.bn3_2(self.conv3_2(x3)), 0.2, inplace=True)
        x3 = F.leaky_relu(self.bn3_3(self.conv3_3(x3)), 0.2, inplace=True)
        # x3 = self.conv3_3(x3)
        x3 = x3.contiguous().view(x3.size(0), -1)
        x3 = F.relu(self.fc1_1(x3), inplace=True)

        x4 = F.leaky_relu(self.bn4_2(self.conv4_2(x4)), 0.2, inplace=True)
        # x4 = self.conv4_2(x4)
        x4 = x4.contiguous().view(x4.size(0), -1)
        x4 = F.relu(self.fc1_2(x4), inplace=True)

        x5 = x5.contiguous().view(x5.size(0), -1)
        x5 = F.relu(self.fc1_3(x5), inplace=True)

        x = torch.cat([x2, x3, x4, x5], dim=1)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.relu(self.fc3(x), inplace=True)
        x = F.tanh(self.fc3(x))

        viewp = F.tanh(self.fc3_6(x))
        y = F.relu(self.fc6(viewp), inplace=True)
        y = F.tanh(self.fc7(y))
        x = torch.cat([x, y], dim=1)
        return x,viewp

class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=0.1, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()

        # self.register_buffer('vertices_base', filename_obj[0].cpu())  # vertices_base)
        # self.register_buffer('faces', filename_obj[1].cpu())  # faces)
        self.nv = 7331 + 738 # max vertex with max wheels
        # self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5
        # dim = 1024
        dim = 1500
        dim_hidden = [dim, dim*2]

        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        # self.fc_centroid = nn.Linear(dim_hidden[1], 30+20)
        self.fc_para = nn.Linear(dim_hidden[1], 30 + 21)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)
        self.fc_c0 = nn.Linear(self.nv*3, 1507*3)
        # self.fc_ccs0 = nn.Linear(5824 * 3, 2280 * 3)
        # self.fc_c1 = nn.Linear(5824 * 3, 3544 * 3)
        self.fc_p0 = nn.Linear(self.nv * 3, (self.nv-1507) * 3)
        # self.fc_pt = nn.Linear(1507 * 3, 956 * 3)
        # self.fc_pg = nn.Linear(1507 * 3, 551 * 3)
        self.fcon_turret1 = nn.Sequential(nn.Linear(956*3, 956*3),
                                         # nn.BatchNorm1d(3),
                                         nn.Tanh())
        self.fcon_gun1 = nn.Sequential(nn.Linear((1507-956)*3, (1507-956)*3),
                                      # nn.BatchNorm1d(3),
                                      nn.Tanh())
        self.fcon_body1 = nn.Sequential(nn.Linear((3787-1507)*3, (3787-1507)*3),
                                       # nn.BatchNorm1d(3),
                                       nn.Tanh())
        self.fcon_guard1 = nn.Sequential(nn.Linear((4425-3787)*3, (4425-3787)*3),
                                        # nn.BatchNorm1d(3),
                                        nn.Tanh())
        self.fcon_track1 = nn.Sequential(nn.Linear((5269-4425)*3, (5269-4425)*3),
                                        # nn.BatchNorm1d(3),
                                        nn.Tanh())
        self.fcon_wheel1 = nn.Sequential(nn.Conv1d(3, 3, 1),
                                        nn.BatchNorm1d(3),
                                        nn.Tanh())

        obj_file = datalocation + '/data/tankpart03/'
        # Load obj file
        tankparts = os.listdir(obj_file)
        meshes, self.verts1s, self.faces1s = [], [], []
        for name in tankparts:
            obj_filename = obj_file + name
            # mesh = load_objs_as_meshes([obj_filename], device=device)
            # meshes.append(mesh)
            verts1, faces1, aux1 = load_obj(obj_filename)
            self.verts1s.append(verts1.to(device))
            self.faces1s.append(faces1)

        duicheng = torch.tensor([-1, 1, 1]).to(device)# Flipped Axis
        # zhongxin = torch.tensor([0.01, 0, 0]).to(device)
        zhongxin = torch.tensor([0.005, 0, 0]).to(device)   #part shape center
        a1, a2 = torch.tensor(-0.093).to(device), torch.tensor(-0.123).to(device) #Body topology connection point
        PG_XZ1 = torch.tensor([0, 0, 0.035]).to(device) # Gun barrel topology connection point
        self.canshu = [duicheng,zhongxin,a1, a2,PG_XZ1]
        # wheel topology connection point
        self.xql, self.yql, self.zql = torch.tensor(-0.263).to(device), torch.tensor(-0.271).to(device), torch.tensor(
            0.041).to(device).to(
            device)
        self.xzl, self.yzl, self.zzl = torch.tensor(-0.12).to(device), torch.tensor(-0.034).to(device), torch.tensor(
            -0.035).to(device).to(device)
        self.xhl, self.yhl, self.zhl = torch.tensor(-0.265).to(device), torch.tensor(0.817).to(device), torch.tensor(
            0.027).to(device)

        # self.luntaishu = torch.tensor(6).to(device)
        # self.weiyi = torch.tensor(0.0).to(device)
        # self.lunweiyi1,self.lunweiyi2 = torch.tensor(0.12).to(device),torch.tensor(0.08).to(device)


    def forward(self, xx):
        batch_size = xx.shape[0]
        xx = F.relu(self.fc1(xx), inplace=True)
        xx = F.relu(self.fc2(xx), inplace=True)

        allparas = self.fc_para(xx)
        para_LS = torch.tensor(0.2) + torch.sigmoid(allparas[:,0:30])*3
        para_WY = torch.tanh(allparas[:,30:60])*0.5

        xx0 = torch.tanh(self.fc_bias(xx)) * 0.9
        fc_c0 = torch.tanh(self.fc_c0(xx0))
        fc_p0 = torch.tanh(self.fc_p0(xx0))
        # bias = torch.cat((fc_c0, fc_p0), dim=1) * 0.3
        bias = torch.cat((fc_c0, fc_p0), dim=1) * 0.9

        out1_turret = self.fcon_turret1(bias[:,  :3*956])
        out1_gun = self.fcon_gun1(bias[:,  3*956:3*1507])
        out1_body = self.fcon_body1(bias[:,  3*1507:3*3787])
        out1_guard = self.fcon_guard1(bias[:,  3*3787:3*4425])
        out1_track = self.fcon_track1(bias[:,  3*4425:3*5269])
        out1_turret = out1_turret.view(batch_size, 3,-1)
        out1_gun = out1_gun.view(batch_size, 3,-1)
        out1_body = out1_body.view(batch_size, 3,-1)
        out1_guard = out1_guard.view(batch_size, 3,-1)
        out1_track = out1_track.view(batch_size, 3,-1)
        bias2 = bias.view(-1, 3,self.nv)
        out1_wheel = self.fcon_wheel1(bias2[:, :, 5269:])

        out4 = torch.cat([out1_turret,out1_gun,out1_body, out1_guard,out1_track,out1_wheel],dim =2)* 0.3

        bias = out4.permute(0,2,1)
        All_vertex, All_faces,indexnode,vertices,wheelnum = [],[],[],[],[]
        # self.lunshu = torch.tensor(4).to(device)
        for i in range(batch_size):
            # self.lunshu += 2
            para_LSi, para_WYi = para_LS[i].unsqueeze(0), para_WY[i].unsqueeze(0)
            allvertex, allfaces, indexnode_ = self.gettank_all_0d5(para_LSi, para_WYi)
            All_vertex.append(allvertex)
            All_faces.append(allfaces)
            indexnode.append(indexnode_[0:8])
            wheelnum.append(indexnode_[8].unsqueeze(0))

            bias_ = bias[i, :allvertex.shape[0], :]
            base0 = torch.tanh(allvertex.unsqueeze(0))
            sign = torch.sign(base0)
            base = torch.abs(base0)
            base = torch.log((base + 0.001) / (1.1 - base)) * 1.1

            vertices_ = torch.sigmoid(base + bias_) * sign * 1
            vertices.append(vertices_[0])

        Mesh_init = Meshes(verts=All_vertex, faces=All_faces)
        Mesh_final = Meshes(verts=vertices, faces=All_faces)
        # indexnode = torch.cat(indexnode,dim = 0)
        wheelnum = torch.cat(wheelnum,dim = 0)
        return Mesh_init, Mesh_final, [indexnode,wheelnum]


    def gettank_all_0d5(self,para_LS, para_WY):
        [duicheng, zhongxin, a1, a2, PG_XZ1] = self.canshu
        CS_LS = para_LS[0][0:3]  # 3个
        CS_WY = para_WY[0][0:3]  # 3个
        # CS1_LS = para_LS[0][3:6]  # 3个
        # CS1_LS = torch.tensor([para_LS[0][3], para_LS[0][1], para_LS[0][2]]).to(device)
        CS1_LS = torch.tensor([para_LS[0][3], para_LS[0][4], para_LS[0][2]]).to(device)
        a3 = -(a1 - a2) * CS1_LS[0] + (a1 * CS_LS[0] - a2 * CS1_LS[0])
        # CS1_WY = torch.tensor([para_WY[0][3], para_WY[0][3], para_WY[0][4]]).to(device)
        CS1_WY = torch.tensor([a3, para_WY[0][3]/4, para_WY[0][4]/4]).to(device)
        # CS1_WY = para_LS[0][3:6]
        # CS1_WY = torch.tensor([a3, para_WY[0][3], para_WY[0][4]]).detach().to(device)  # 2个
        # # CS1_WY2 = torch.cat((a3, para_WY[0][3], para_WY[0][4]))  # 2个
        # # CS1_WY2 = torch.cat([a3, para_WY[0][3], para_WY[0][4]],dim = 0)
        # LD_LS = torch.tensor([para_LS[0][3], para_LS[0][7], para_LS[0][5]]).detach().to(device)  # 1和之前一样共1个  第三个第一个一样
        # LD_WY = torch.tensor([a3, para_WY[0][5], para_WY[0][6]]).to(device)  # 共2个，且移动限制很大
        # LD_LS = torch.tensor([para_LS[0][3], para_LS[0][1], para_LS[0][2]]).detach().to(device)  # 1和之前一样共1个  第三个第一个一样
        # LD_WY = torch.tensor([a3, 0, 0]).to(device)  # 共2个，且移动限制很大
        LD_LS = torch.tensor([para_LS[0][3], para_LS[0][4], para_LS[0][2]]).detach().to(device)  # 1和之前一样共1个  第三个第一个一样
        LD_WY = torch.tensor([a3, para_WY[0][4]/4, para_WY[0][4]/4]).to(device)  # 共2个，且移动限制很大
        #
        # HL_LS = torch.tensor([para_LS[0][8], para_LS[0][8], para_LS[0][8]]).to(device)  # 1个，倍数，是一个数
        HL_LS = LD_LS
        # a_houlun, y, z = torch.tensor(-0.12).to(device), torch.tensor(0.411).to(device), torch.tensor(0.044).to(device)
        ax_hl = -(a1 - self.xhl) * HL_LS[0] + (a1 * CS_LS[0] - self.xhl * HL_LS[0])
        ay_hl = (LD_LS[2] - HL_LS[2]) * self.yhl
        az_hl = (LD_LS[1] - HL_LS[1]) * self.zhl
        # HL_WY = torch.tensor([ax_hl, az_hl + LD_WY[1], -1 * ay_hl + LD_WY[2]]).to(device)
        # QL_LS = torch.tensor([para_LS[0][9], para_LS[0][9], para_LS[0][9]]).to(device)
        HL_WY = LD_WY
        QL_LS = LD_LS
        ax_hl = (CS_LS[0] - QL_LS[0]) * self.xql
        ay_hl = -1 * (LD_LS[2] - QL_LS[2]) * self.yql
        az_hl = (LD_LS[1] - QL_LS[1]) * self.zql
        # QL_WY = torch.tensor([ax_hl, az_hl + LD_WY[1], ay_hl + LD_WY[2]]).to(device)
        QL_WY = LD_WY
        ZL_LS = torch.tensor([0.7*para_LS[0][10], 0.7*para_LS[0][10], 0.7*para_LS[0][10]]).to(device)
        ax_hl = (LD_LS[0] - ZL_LS[0]) * self.xzl + LD_WY[0]
        ay_hl = -1 * (LD_LS[2] - QL_LS[2]) * self.yzl + LD_WY[2]
        az_hl = (LD_LS[1] - ZL_LS[1]) * self.zzl + LD_WY[1]
        ZL_WY = torch.tensor([ax_hl, az_hl, ay_hl]).to(device)
        # para_LS[0][11:16] = 0.02 * para_LS[0][11:16]

        lunshu = (torch.floor(para_LS[0][11]*5)).int()
        lunshu = max(lunshu, 4)
        lunshu = min(lunshu, 9)
        alld = torch.tensor(0.).to(device)
        for i in range(lunshu-1):
            alld += torch.abs(para_WY[0][12+i])
        lunweiyi = []
        for i in range(lunshu-1):
            weiyi =torch.abs(0.32 * LD_LS[2]/alld * para_WY[0][12+i])
            lunweiyi.append(weiyi)
        lunweiyi = 1 * torch.tensor(lunweiyi).to(device)
        # alld = para_LS[0][11] + para_LS[0][12]+para_LS[0][13]+para_LS[0][14]+para_LS[0][15]
        # self.lunweiyi1 = 0.32 * LD_LS[2]/alld * para_LS[0][11]
        # self.lunweiyi2 = 0.32 * LD_LS[2]/alld * para_LS[0][12]
        # self.lunweiyi3 = 0.32 * LD_LS[2]/alld * para_LS[0][13]
        # self.lunweiyi4 = 0.32 * LD_LS[2]/alld * para_LS[0][14]
        # self.lunweiyi5 = 0.32 * LD_LS[2]/alld * para_LS[0][15]
        # lunweiyi = 1 * torch.tensor([self.lunweiyi1,self.lunweiyi2,self.lunweiyi3,self.lunweiyi4,self.lunweiyi5]).to(device)

        PT_LS = torch.tensor([para_LS[0][17] * 1.0, para_LS[0][18] * 1.5, para_LS[0][19] * 1.0]).to(device)  # 3个
        z = torch.tensor(0.084).to(device)
        az = (CS_LS[1] - PT_LS[1]) * z
        PT_WY = torch.tensor([para_WY[0][7], az, para_WY[0][8]]).to(device)  # 共2个
        # PT_WY = torch.tensor([para_WY[0][7], para_WY[0][12], para_WY[0][8]]).to(device)  # 共2个

        # PG_LS = torch.tensor([para_LS[0][20]*4, para_LS[0][21]*4, para_LS[0][22] ** 2]).to(device)  # 3个
        PG_LS = torch.tensor([para_LS[0][20] * 1.5, para_LS[0][21] * 1.5, para_LS[0][22] * 1.0]).to(device)  # 3个
        # x, y, z = torch.tensor(0.005).to(device), torch.tensor(0.026).to(device), torch.tensor(0.118).to(device)
        x, y, z = torch.tensor(0.005).to(device), torch.tensor(0.034).to(device), torch.tensor(0.121).to(device)
        ax_hl = (PT_LS[0] - PG_LS[0]) * x + PT_WY[0]
        ay_hl = -1 * (PT_LS[2] - PG_LS[2]) * y + PT_WY[2]
        az_hl = (PT_LS[1] - PG_LS[1]) * z + PT_WY[1]
        PG_WY = torch.tensor([ax_hl, az_hl, ay_hl]).to(device)
        #
        XH_LS = torch.tensor([para_LS[0][23], para_LS[0][24], para_LS[0][25]]).to(device)  # 3个
        z = torch.tensor(0.14).to(device)
        az = (PT_LS[1] - XH_LS[1]) * z + PT_WY[1]
        XH_WY = torch.tensor([para_WY[0][9], az, para_WY[0][10]]).to(device)  # 2个
        #
        jiao = torch.abs(para_WY[0][11] *0.5)  # 1个
        # jiao = 0.0
        # PG_XZ1 = torch.tensor([0, 0, -0.017]).to(device)
        PG_XZ2 = torch.tensor([[1, 0, 0],
                               [0, 1, 0],
                               [0, jiao, 1]]).to(device)
        all_LS = para_LS[0][26:29]

        a = 0

        vertextemp = self.verts1s[5] * PT_LS + PT_WY
        # allvertex = torch.cat([allvertex, vertextemp], dim=0)
        # face_idx2 = (self.faces1s[5].verts_idx + a).to(device)
        # allfaces = torch.cat([allfaces, face_idx2], dim=0)
        allvertex, allfaces = vertextemp, self.faces1s[5].verts_idx.to(device)
        a += len(vertextemp)
        # a2 = a
        # c2 = len(allfaces)
        a0 = a
        c0 = len(allfaces)

        vertextemp = self.verts1s[4] * PG_LS + PG_WY
        piancha  = PG_XZ1* PG_LS + PG_WY
        vertextemp = vertextemp - piancha
        vertextemp = torch.matmul(vertextemp, PG_XZ2) + piancha
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[4].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)
        a1 = a
        c1 = len(allfaces)
        # vertextemp = self.verts1s[7] * XH_LS + XH_WY
        # allvertex = torch.cat([allvertex, vertextemp], dim=0)
        # face_idx2 = (self.faces1s[7].verts_idx + a).to(device)
        # allfaces = torch.cat([allfaces, face_idx2], dim=0)
        # a += len(vertextemp)

        vertextemp = self.verts1s[0] * CS_LS
        # vertextemp = self.verts1s[0] * para_LS[0][3]
        # allvertex, allfaces = vertextemp, self.faces1s[0].verts_idx.to(device)
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[0].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)

        vertextemp = self.verts1s[1] * CS1_LS + CS1_WY
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[1].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)
        vertextemp = vertextemp * duicheng + zhongxin
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[1].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)

        a2 = a
        c2 = len(allfaces)

        vertextemp = self.verts1s[3] * LD_LS + LD_WY
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[3].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)

        vertextemp = vertextemp * duicheng + zhongxin
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[3].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)

        b = 0
        vertextemp = self.verts1s[2] * HL_LS + HL_WY
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[2].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        all_lunv, all_lunf = vertextemp, self.faces1s[2].verts_idx.to(device)
        b += len(vertextemp)
        a += len(vertextemp)

        vertextemp = self.verts1s[6] * QL_LS + QL_WY
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[6].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)
        all_lunv = torch.cat([all_lunv, vertextemp], dim=0)
        all_lunf = torch.cat([all_lunf, (self.faces1s[6].verts_idx + b).to(device)], dim=0)
        b += len(vertextemp)

        vertextemp = self.verts1s[8] * ZL_LS + ZL_WY
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (self.faces1s[8].verts_idx + a).to(device)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)
        all_lunv = torch.cat([all_lunv, vertextemp], dim=0)
        all_lunf = torch.cat([all_lunf, (self.faces1s[8].verts_idx + b).to(device)], dim=0)
        b += len(vertextemp)

        self.weiyi = torch.tensor(0.0).to(device)
        for i in range(lunshu- 1):
        # for i in range(self.luntaishu - 1):
            self.weiyi += lunweiyi[i]
            vertextemp0 = vertextemp - torch.tensor([0, 0, self.weiyi]).to(device)
            allvertex = torch.cat([allvertex, vertextemp0], dim=0)
            face_idx2 = (self.faces1s[8].verts_idx + a).to(device)
            allfaces = torch.cat([allfaces, face_idx2], dim=0)
            a += len(vertextemp0)
            all_lunv = torch.cat([all_lunv, vertextemp0], dim=0)
            all_lunf = torch.cat([all_lunf, (self.faces1s[8].verts_idx + b).to(device)], dim=0)
            b += len(vertextemp0)

        vertextemp = all_lunv * duicheng + zhongxin
        allvertex = torch.cat([allvertex, vertextemp], dim=0)
        face_idx2 = (all_lunf + a)
        allfaces = torch.cat([allfaces, face_idx2], dim=0)
        a += len(vertextemp)
        a3 = a
        c3 = len(allfaces)

        allvertex = allvertex * all_LS + CS_WY
        return allvertex, allfaces,[a0,a1,a2,a3,c0,c1,c2,c3,lunshu]

    def test_gettenk_all(self):

        para_LS = torch.full([1, 30], 2.00, device=device, requires_grad=True)
        para_WY = torch.full([1, 30], 0.0, device=device, requires_grad=True)
        allvertex, allfaces, [a1, a2, a3, c1, c2] = self.gettank_all_0d5(para_LS, para_WY)
        final_obj = 'CHESHEN000.obj'
        save_obj(final_obj, allvertex, allfaces)

        

class Model_torch3d(nn.Module):
    def __init__(self, filename_obj=None, image_size = 256):
        super(Model_torch3d, self).__init__()

        # auto-encoder
        self.encoder = Encoder(im_size=image_size)
        self.decoder = Decoder(filename_obj)

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def forward(self, images=None,view = None):
        res,viewp = self.encoder(images,view)
        Mesh_init, Mesh_final, indexnode = self.decoder(res)
        return Mesh_init, Mesh_final, indexnode, viewp
        # vertices, faces, nodes = self.decoder(res)
        # return vertices, faces, nodes,viewp
    def test_gettenl(self):
        self.decoder.test_gettenk_all()

    def SaveNet(self,path,epoch):
        torch.save(self.encoder.state_dict(), path + '/encoder_%d.pth' % (epoch + 1))
        torch.save(self.decoder.state_dict(), path + '/decoder_%d.pth' % (epoch + 1))
