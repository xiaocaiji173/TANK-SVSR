import numpy as np
import pickle
import os, sys
from train_parameter import *
from PIL import Image
import random
import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
# from pytorch3d.structures import Textures
import matplotlib.pyplot as plt

faces_per_pixel = 5

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

# word_idx = {'02691156': 0, # airplane
#             '03636649': 1, # lamp
#             '03001627': 2} # chair
#
# idx_class = {0: 'airplane', 1: 'lamp', 2: 'chair'}


class TankShapeNet(data.Dataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.
    Arguments:
    """
    # def __init__(self, file_root, file_list,obj_root,sigma = 7e-6):
    def __init__(self, file_root, obj_root, sigma=1e-5,image_size = 256,allindex = [1,3,8,12,13],device = torch.device("cuda:0")):
    # def __init__(self, file_root, obj_root, sigma=7e-6, image_size=256, allindex=[1, 3, 8, 12, 13],
    #                  device=torch.device("cuda:0")):

        self.file_root = file_root
        # self.obj_root = obj_root
        # Read file list
        with open(obj_root, "r") as fp:
            # self.file_names = fp.read().split("\n")[:-1]
            self.file_names = fp.read().split("\n")
        self.file_nums = len(self.file_names)
        self.image_size = image_size
        self.device = device

        #设定渲染器
        self.num_views = 20
        # elev = torch.linspace(0, 360, self.num_views)
        # azim = torch.linspace(-180, 180, self.num_views)
        # elev = [0, 0, 0, 0, 90, 180, 270, 360, 60, 120, 60, 120, 30, 30, 240, 300, 30, 120, 0, 0]  # 纵向
        # elev = torch.from_numpy(np.array(elev))
        # azim = [-120, -90, 120, 90, 0, 0, 0, 0, -30, -30, 30, 30, 120, -120, 60, 60, 180, 180, 30, -45]  # 横向
        elev = [0, 0, 0, 0, 90, 360, 160, 180, 60, 120, 60, 120, 30, 30, 45, 60, 120, 30, 0, 0]  # 纵向
        elev = torch.from_numpy(np.array(elev))
        azim = [-135, -90, 135, 90, 0, 0, 0, 0, -30, -30, 30, 30, 120, -120, 60, 60, 180, 180, 30, -45]  # 横向
        azim = torch.from_numpy(np.array(azim))
        self.lights = DirectionalLights(device=self.device, direction=[[1.0, 1.0, -1.0]])
        # self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        self.R, self.T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        self.cameras = FoVOrthographicCameras(device=self.device, R=self.R, T=self.T)
        # self.cameras = OpenGLPerspectiveCameras(device=self.device, R=self.R, T=self.T)
        R2, T2 = look_at_view_transform(dist=2.7, elev=elev[3:6], azim=azim[3:6])
        # self.part_cameras = OpenGLPerspectiveCameras(device=self.device, R=R2, T=T2)
        self.part_cameras = FoVOrthographicCameras(device=self.device, R=R2, T=T2)
        # self.part_cameras = OpenGLPerspectiveCameras(device=self.device, R=self.R[None, 3:6, ...],
        #                                   T=self.T[None, 3:6, ...])

        # camera = OpenGLPerspectiveCameras(device=self.device, R=self.R[None, 1, ...],
        #                                   T=self.T[None, 1, ...])
        camera = FoVOrthographicCameras(device=self.device, R=self.R[None, 1, ...],
                                        T=self.T[None, 1, ...])
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            # image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
            shader=SoftPhongShader(
            device=self.device,
            cameras=camera,
            lights=self.lights
        ))

        # sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=faces_per_pixel,
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        self.allindex = allindex

        viwe0 = self.R.reshape(-1, 9)
        self.view = torch.cat([viwe0, self.T], dim=1)
        # a = view.cpu().numpy()
        self.lighttensor = torch.tensor([[0.0, 0.0, -3.0]])

    def __getitem__(self, index):
        viewno = random.choice(self.allindex)
        viewno = 3
        objname = os.path.join(self.file_root, self.file_names[index])
        pathlist = self.file_names[index].split('/')
        objpath = os.path.join(self.file_root, pathlist[0])
        print(pathlist[0])
        # obj_filename = os.path.join(filename_obj, objfile_name)
        # vertes, faces, aux = load_obj(obj_filename, device=self.device)
        mesh = load_objs_as_meshes([objname], device=self.device)
        mesh = mesh.scale_verts(vertexscale)
        # vertes, faces = mesh.get_mesh_verts_faces(0)
        meshes = mesh.extend(self.num_views)
        target_images = self.renderer(meshes, cameras=self.cameras, lights=self.lights)
        # target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]
        # target_cameras = [OpenGLPerspectiveCameras(device=self.device, R=self.R[None, i, ...],
        #                                            T=self.T[None, i, ...]) for i in range(self.num_views)]
        # # RGB images
        # image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
        # plt.show()
        silhouette_images = self.renderer_silhouette(meshes, cameras=self.cameras, lights=self.lights)
        # target_silhouette = [silhouette_images[i, ..., 3] for i in range(self.num_views)]
        # # Visualize silhouette images
        # image_grid(silhouette_images.cpu().numpy(), rows=4, cols=5, rgb=False)
        # plt.show()
        # print(pathlist[0])
        allpartimg_ill = self.getpartobj(objpath, cameras=self.cameras)
        img = target_images[viewno,:,:,:3]

        name2 = pathlist[0]+ '_' + str(viewno).rjust(2,'0')
        objpath = os.path.join(self.file_root, pathlist[0])
        wheelnumb = self.getwheelnumb(objpath)
        viewp = torch.cat([self.view[viewno].unsqueeze(0),self.lighttensor],dim = 1)
        return [img,allpartimg_ill,viewno], mesh, name2, target_images, silhouette_images,wheelnumb,viewp



    def __len__(self):
        return self.file_nums

    #get wheel number from txt
    def getwheelnumb(self,path):
        allfile = os.listdir(path)
        for file in allfile:
            if file[-3:]=='txt':
                numb = int(file[:2])
        return numb

    # read image
    def Readimg(self, name):

        C, H, W = 3, self.image_size, self.image_size
        img = Image.open(name)
        img = img.resize((W, H), Image.ANTIALIAS)
        # plt.imshow(img)
        # plt.show()
        img = np.array(img)

        return img

    # get .obj files
    def findobjpaths(self,partpath):
        partobjfiles = os.listdir(partpath)
        result = []
        for pahts in partobjfiles:
            if pahts[-3:] == 'obj':
                result.append(pahts)
        return result

    # get .obj files of different parts
    def getpartobj(self,filename_obj,cameras):
        partpath = filename_obj +  '\\part\\'
        # partobjfiles = os.listdir(partpath)
        allpartobjfiles=self.findobjpaths(partpath)
        mesh1 = load_objs_as_meshes([partpath + allpartobjfiles[1]], device=self.device)
        mesh2 = load_objs_as_meshes([partpath + allpartobjfiles[2]], device=self.device)
        # mesh3 = load_objs_as_meshes([partpath + partobjfiles[5]], device=self.device)
        mesh3 = load_objs_as_meshes([partpath + allpartobjfiles[0]], device=self.device)
        mesh1 = mesh1.scale_verts(vertexscale)
        mesh2 = mesh2.scale_verts(vertexscale)
        mesh3 = mesh3.scale_verts(vertexscale)
        # mesh4 = load_objs_as_meshes([partpath + partobjfiles[10]], device=self.device)
        # camerasnum = 3
        camerasnum = len(self.cameras)
        allsilhouette_images = []
        meshes = mesh1.extend(camerasnum)
        # target_images1 = self.renderer(meshes, cameras=self.part_cameras, lights=self.lights)
        # silhouette_images1 = self.renderer_silhouette(meshes, cameras=self.part_cameras, lights=self.lights)
        # target_images1 = self.renderer(meshes, cameras=cameras, lights=self.lights)
        silhouette_images1 = self.renderer_silhouette(meshes, cameras=cameras, lights=self.lights)
        allsilhouette_images.append(silhouette_images1.unsqueeze(0))

        meshes = mesh2.extend(camerasnum)
        # target_images2 = self.renderer(meshes, cameras=self.part_cameras, lights=self.lights)
        # silhouette_images2 = self.renderer_silhouette(meshes, cameras=self.part_cameras, lights=self.lights)
        # target_images2 = self.renderer(meshes, cameras=cameras, lights=self.lights)
        silhouette_images2 = self.renderer_silhouette(meshes, cameras=cameras, lights=self.lights)
        allsilhouette_images.append(silhouette_images2.unsqueeze(0))

        meshes = mesh3.extend(camerasnum)
        # target_images3 = self.renderer(meshes, cameras=self.part_cameras, lights=self.lights)
        # silhouette_images3 = self.renderer_silhouette(meshes, cameras=self.part_cameras, lights=self.lights)
        # target_images3 = self.renderer(meshes, cameras=cameras, lights=self.lights)
        silhouette_images3 = self.renderer_silhouette(meshes, cameras=cameras, lights=self.lights)
        allsilhouette_images.append(silhouette_images3.unsqueeze(0))

        # meshes = mesh4.extend(camerasnum)
        # target_images3 = self.renderer(meshes, cameras=self.part_cameras, lights=self.lights)
        # silhouette_images3 = self.renderer_silhouette(meshes, cameras=self.part_cameras, lights=self.lights)
        # target_images4 = self.renderer(meshes, cameras=cameras, lights=self.lights)
        # silhouette_images4 = self.renderer_silhouette(meshes, cameras=cameras, lights=self.lights)
        # allsilhouette_images1.append(silhouette_images4)

        # return [target_images1,target_images2,target_images3,target_images4], [silhouette_images1,silhouette_images2,silhouette_images3,silhouette_images4]
        allsilhouette_images = torch.cat(allsilhouette_images,dim = 0)
        return allsilhouette_images


if __name__ == "__main__":

    pass