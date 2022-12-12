import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Resize
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
from pytorch3d.structures import join_meshes_as_batch
import torch
from pytorch3d.structures import Meshes
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
from train_parameter import *

def my_collate(batch):
    # [img, allpartimg_ill, viewno], vertes, faces, name2, target_images, silhouette_images, wheelnumb
    data0 = [item[0] for item in batch]
    mesh = [item[1] for item in batch]
    mesh = join_meshes_as_batch(mesh)
    name2 = [item[2] for item in batch]
    target_images = [item[3].unsqueeze(0) for item in batch]
    silhouette_images = [item[4].unsqueeze(0) for item in batch]
    wheelnumb = [item[5] for item in batch]
    viewp = [item[6] for item in batch]
    wheelnumb = torch.tensor(wheelnumb)
    target_images = torch.cat(target_images,dim=0)
    silhouette_images = torch.cat(silhouette_images, dim=0)
    viewp = torch.cat(viewp, dim=0)

    img = [item[0].unsqueeze(0) for item in data0]
    img = torch.cat(img,dim=0)
    allpartimg_ill = [item[1].unsqueeze(0) for item in data0]
    viewno = [item[2] for item in data0]
    allpartimg_ill = torch.cat(allpartimg_ill,dim=0)

    return [img, allpartimg_ill, viewno], mesh, name2, target_images, silhouette_images, wheelnumb,viewp


def getrender(device, num_views, sigma=1e-5,imagesize = 256):
    # sigma = 1e-5
    # elev = torch.linspace(0, 360, num_views)
    # azim = torch.linspace(-180, 180, num_views)
    elev = [0, 0, 0, 0, 90, 360, 160, 180, 60, 120, 60, 120, 30, 30, 45, 60, 120, 30, 0, 0]  # 纵向
    elev = torch.from_numpy(np.array(elev))
    azim = [-135, -90, 135, 90, 0, 0, 0, 0, -30, -30, 30, 30, 120, -120, 60, 60, 180, 180, 30, -45]  # 横向
    azim = torch.from_numpy(np.array(azim))
    lights = DirectionalLights(device=device, direction=[[1.0, 1.0, -1.0]])

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    # lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    # camera = OpenGLPerspectiveCameras(device=device, R=R[None, 3, ...],
    #                                   T=T[None, 3, ...])
    camera = FoVOrthographicCameras(device=device, R=R[None, 3, ...], T=T[None, 3, ...])

    R2, T2 = look_at_view_transform(dist=2.7, elev=elev[3:6], azim=azim[3:6])
    # part_cameras = OpenGLPerspectiveCameras(device=device, R=R2, T=T2)
    part_cameras = FoVOrthographicCameras(device=device, R=R2, T=T2)

    raster_settings_soft = RasterizationSettings(
        image_size=imagesize,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
    )
    raster_settings = RasterizationSettings(
        image_size=imagesize,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device,
                               cameras=camera,
                               lights=lights)
    )
    # target_cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...],
    #                                            T=T[None, i, ...]) for i in range(num_views)]
    target_cameras = [FoVOrthographicCameras(device=device, R=R[None, i, ...],
                                             T=T[None, i, ...]) for i in range(num_views)]

    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )
    viwe0 = R.reshape(-1,9)
    view = torch.cat([viwe0,T],dim=1)
    # a = view.cpu().numpy()
    lighttensor = torch.tensor([[0.0, 0.0, -3.0]]).to(device)
    return  renderer_textured,renderer_silhouette,target_cameras,lights,view, lighttensor,part_cameras

# Show a visualization comparing the rendered predicted mesh to the ground truth
# mesh
# def visualize_prediction(predicted_mesh, renderer = renderer_silhouette,target_image = target_rgb[1], title='',silhouette=False):
def visualize_prediction(predicted_mesh, renderer, target_image, title='',silhouette=False,filepath = "filename.png"):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    # plt.show()

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    # plt.show()
    plt.savefig(filepath)

def calpartloss(src_mesh, allino,sil_images,Bsize,partindex_n = 2):
    [sil_images_cs,sil_images_pg,sil_images_pt] = sil_images
    allpt_vertex,allpt_face,allpg_vertex, allpg_face,allcs_vertex, allcs_face = [],[],[],[],[],[]
    # silhouette_images1,silhouette_images2,silhouette_images3 = [],[],[]
    for i in range(Bsize):
        ino = allino[i]
        new_vertes, new_face = src_mesh.get_mesh_verts_faces(i)
        pt_vertex,pt_face = new_vertes[:ino[0]],new_face[:ino[4]]
        pg_vertex, pg_face = new_vertes[ino[0]:ino[1]], new_face[ino[4]:ino[5]] - ino[0]
        cs_vertex, cs_face = new_vertes[ino[1]:], new_face[ino[5]:] - ino[1]

        allpt_vertex.append(pt_vertex)
        allpt_face.append(pt_face)
        allpg_vertex.append(pg_vertex)
        allpg_face.append(pg_face)
        allcs_vertex.append(cs_vertex)
        allcs_face.append(cs_face)
    mesh_pt = Meshes(verts=allpt_vertex, faces=allpt_face)
    mesh_pg = Meshes(verts=allpg_vertex, faces=allpg_face)
    mesh_cs = Meshes(verts=allcs_vertex, faces=allcs_face)

    randomviewlist0 = np.random.permutation(randomviewlist)[:partindex_n]
    randomviewlist1 = np.append(np.array([3, 4, 5]), randomviewlist0)

    partloss = torch.tensor(0.).to(device)
    a,b,c = 0.5,0.5,0.8
    for j in randomviewlist1:
    # for j in np.random.permutation(num_views).tolist()[:index_n]:
        silhouette_images_predicted_cs = renderer_silhouette(mesh_cs, cameras=target_cameras[j], lights=lights)
        silhouette_images_predicted_pg = renderer_silhouette(mesh_pg, cameras=target_cameras[j], lights=lights)
        silhouette_images_predicted_pt = renderer_silhouette(mesh_pt, cameras=target_cameras[j], lights=lights)
        if j==3:
            a,b,c = 0.1,0.1,0.1
        partloss += a * (torch.abs(silhouette_images_predicted_cs[:, :, :, 3] - sil_images_cs[:, j, :, :, 3])).mean()/(partindex_n+1)
        partloss += b * (torch.abs(silhouette_images_predicted_pg[:, :, :, 3] - sil_images_pg[:, j, :, :, 3])).mean()/(partindex_n+1)
        partloss += c * (torch.abs(silhouette_images_predicted_pt[:, :, :, 3] - sil_images_pt[:, j, :, :, 3])).mean()/(partindex_n+1)
        # visualize_img(silhouette_images_predicted_cs, sil_images_cs[:, j, :, :, :], silhouette=True)
        # visualize_img(silhouette_images_predicted_pt, sil_images_pt[:, j, :, :, :], silhouette=True)
        # visualize_img(silhouette_images_predicted_pg, sil_images_pg[:, j, :, :, :], silhouette=True)
    return partloss

def set_requires_grad(nets, requires_grad=False,selfA=0,self_grade=False,ifself=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    if ifself:
        for param in selfA.parameters():
            param.requires_grad = self_grade

def getDlabel(bsize):
    reallabel = torch.ones(bsize).clone().float().detach().reshape(bsize, -1)
    fakelabel = torch.zeros(bsize).clone().float().detach().reshape(bsize, -1)
    return reallabel,fakelabel

def visualize_img(predicted_images, target_image, title='',silhouette=False,filepath = "filename.png"):
    inds = 3 if silhouette else range(3)
    for i in range(len(predicted_images)):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_images[i, ..., inds].cpu().detach().numpy())
        # plt.show()
        plt.subplot(1, 2, 2)
        plt.imshow(target_image[i, ..., inds].cpu().detach().numpy())
        plt.title(title)
        plt.axis("off")
        plt.show()
    # plt.savefig(filepath)

renderer_textured,renderer_silhouette,target_cameras,lights,view, lighttensor,part_cameras = getrender(device,num_views,sigma=sigma_re,imagesize = imagesize)