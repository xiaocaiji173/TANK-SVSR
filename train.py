from TankShapeNet import TankShapeNet
from pytorch3d.ops import sample_points_from_meshes
from models_large import Model_torch3d
import discriminator
import random
from train_utils import *
import showresult

from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
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
import perceptualloss

def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

    # Show a visualization comparing the rendered predicted mesh to the ground truth
    # mesh
    def visualize_prediction(predicted_mesh, renderer=renderer_silhouette,
                             target_image=target_rgb[1], title='',
                             silhouette=False):
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
        plt.show()
        plt.figure().clear()
        plt.close()

dataset = TankShapeNet(objRoot, objTrainList,image_size=imagesize)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = shuffle_ , num_workers=int(workers),collate_fn=my_collate)
sphere_verts_rgb = torch.full([batchsize, 8100, 3], 0.5, device=device, requires_grad=False)
PerceptualLoss = perceptualloss.PerceptualLoss(requires_grad=False).to(device)
# loop = tqdm(range(nEpoch))

#define model
model = Model_torch3d()
model = model.to(device)
Discriminator =discriminator.Discriminator()
Discriminator = Discriminator.to(device)
BCE_loss = torch.nn.BCELoss().to(device)
L1_loss = torch.nn.L1Loss().to(device)

# optimizer = torch.optim.Adam(model.model_param(), lr=0.00010)
# optimizer=torch.optim.Adam([{"params":model.model_param()},{"params":sphere_verts_rgb}],
#                            lr=0.00010)
optimizer=torch.optim.Adam([{"params":model.model_param()}],lr=0.00010)
optimizerD=torch.optim.Adam(Discriminator.parameters(),lr=0.00010)
# optimizer = torch.optim.SGD(model.model_param(), lr=0.00020, momentum=0.9)

for epoch in range(nEpoch):

    for i, data in enumerate(dataloader, 0):
        imgs, tarmeshes, tankpath, target_images, silhouette_images, wheelnumb,viewp = data
        [img, allpartimg_ill_all, index] = imgs
        Bsize = img.size()[0]
        reallabel, fakelabel = getDlabel(Bsize)
        reallabel, fakelabel = reallabel.to(device), fakelabel.to(device)
        wheelnumb = wheelnumb.to(device)
        image = img[:, :, :, :3].permute(0, 3, 1, 2).to(device)
        view0 = viewp.to(device)
        allpartimg_ill = allpartimg_ill_all.to(device)  # batch,partno,viewno,img,img,img
        sil_images_pg, sil_images_pt, sil_images_cs = allpartimg_ill[:,0,:, :, :, :],allpartimg_ill[:,1,:, :, :, :],\
                                                      allpartimg_ill[:,2,:, :, :, :]

        target_silhouette = [silhouette_images[:,m, ..., 3] for m in range(num_views)]
        target_rgb = [target_images[:,m, ..., :3] for m in range(num_views)]
        trg_mesh = tarmeshes

        #train D
        set_requires_grad(Discriminator, True)
        optimizerD.zero_grad()
        Dlossf,Dlossr,Dlossp = torch.tensor(0.0).to(device),torch.tensor(0.0).to(device),torch.tensor(0.0).to(device)

        Mesh_init, Mesh_final, [indexnode,wheelnum], viewp = model(image, view0)
        new_src_mesh = Mesh_final
        temp_mesh = Mesh_init
        # sphere_verts_rgb_ = []
        # [new_vertes, temp_vertex], new_face, [indexnode,wheelnum],viewp = model(image, view0)
        # new_src_mesh = Meshes(verts=[new_vertes[0]], faces=[new_face[0]])
        # temp_mesh = Meshes(verts=[temp_vertex], faces=[new_face[0]])

        # target_mesh = sample_points_from_meshes(trg_mesh, 7000)
        sphere_verts_rgb_ = [sphere_verts_rgb[i, :indexnode[i][3], :] for i in range(Bsize)]
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb_)
        # for i in range(Bsize):
        #     new_src_mesh[i].textures = TexturesVertex(verts_features=
        #                                            sphere_verts_rgb[i, :indexnode[i][3], :])

        for j in [3,4,5]:
            # new_src_mesh = trg_mesh[0].extend(2)
            silhouette_images_predicted = renderer_silhouette(new_src_mesh, cameras=target_cameras[j], lights=lights)
            predicted_silhouette = silhouette_images_predicted[..., 3]
            # loss_silhouette = (torch.abs(predicted_silhouette - target_silhouette[j])).mean()
            a = predicted_silhouette.unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
            b = target_silhouette[j].unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
            out1, out4, out6, out8, out10 = Discriminator(a, b)
            out1r, out4r, out6r, out8r, out10r = Discriminator(b, b)
            Dlossf += BCE_loss(out10,fakelabel)/3
            Dlossr += BCE_loss(out10, reallabel) / 3
            Dlossp += (discriminator.get_Perception(out1r,out1,2.5) + discriminator.get_Perception(out4r,out4,1.5) +
                      discriminator.get_Perception(out6r,out6,1.5) + discriminator.get_Perception(out8r,out8,1.5))/3
        Dloss = torch.mean(Dlossf+Dlossr) + torch.max((torch.tensor(1.5).to(device)-Dlossp),torch.tensor(0.0).to(device))
        Dloss.backward(retain_graph = True)
        optimizerD.step()

        #train G
        # weigh_3 = 1.5
        set_requires_grad(Discriminator, False)
        optimizer.zero_grad()
        Mesh_init, Mesh_final, [indexnode,wheelnum], viewp = model(image, view0)
        wheelpre = wheelnum.float()
        ino = indexnode
        new_src_mesh = Mesh_final
        temp_mesh = Mesh_init
        # sphere_verts_rgb_ = [sphere_verts_rgb[i, :indexnode[i][3], :] for i in range(Bsize)]
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb_)
        # a = new_src_mesh.get_mesh_verts_faces(0)
        # # vertes, faces = new_src_mesh.get_mesh_verts_faces(0)
        # # sphere_verts_rgb_ = []
        # new_src_mesh = Meshes(verts=[new_vertes[0]], faces=[new_face[0]])
        # temp_mesh = Meshes(verts=[temp_vertex[0]], faces=[new_face[0]])
        target_mesh = sample_points_from_meshes(trg_mesh, 7000)
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        # f2 = len(new_face[0]) - ino[6]
        # f1 = f2 - ino[5]
        # f0 = f1 - ino[4]
        silimages = [sil_images_cs,sil_images_pg,sil_images_pt]
        partloss = calpartloss(new_src_mesh,ino,silimages,Bsize,partindex_n)
        partloss2 = calpartloss(temp_mesh,ino,silimages,Bsize,partindex_n)
        partloss += partloss2 * 0.01
        loss["silhouette"] += partloss

        randomviewlist0 = np.random.permutation(randomviewlist)[:num_views_per_iteration]
        # for j in randomviewlist0:
        randomviewlist1 = np.append(np.array([3, 4, 5]), randomviewlist0)
        Gloss, Glossp = torch.tensor(0.0).to(device),torch.tensor(0.0).to(device)
        for j in randomviewlist1:
            silhouette_images_predicted = renderer_silhouette(new_src_mesh, cameras=target_cameras[j], lights=lights)

            predicted_silhouette = silhouette_images_predicted[..., 3]
            # loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss_silhouette = (torch.abs(predicted_silhouette - target_silhouette[j])).mean()
            # if epoch==0 and j ==3:
            #     visualize_img(silhouette_images_predicted, silhouette_images[:,j, ..., :], silhouette=True,title =tankpath[0] )

            a = predicted_silhouette.unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
            b = target_silhouette[j].unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
            if j<3:
                out1, out4, out6, out8, out10 = Discriminator(a, b)
                out1r, out4r, out6r, out8r, out10r = Discriminator(b, b)
                Gloss += BCE_loss(out10, reallabel)
                Glossp += (discriminator.get_Perception(out1r, out1, 2.5) + discriminator.get_Perception(out4r, out4, 1.5) +
                       discriminator.get_Perception(out6r, out6, 1.5) + discriminator.get_Perception(out8r, out8, 1.5))

            PLoss = 0.001 * PerceptualLoss(a, b)
            # PLoss = torch.max(Glossp,torch.tensor(0.).to(device)) * 0.001
            loss["silhouette"] += (PLoss + 0.005 * Gloss) / num_views_per_iteration
            loss["silhouette"] += loss_silhouette / (num_views_per_iteration)

            # predicted_rgb = images_predicted[..., :3]
            # loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
            # loss["rgb"] += loss_rgb / num_views_per_iteration
        sample_src = sample_points_from_meshes(new_src_mesh, 7000)
        loss["chamfer"], _ = chamfer_distance(target_mesh, sample_src)
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        sum_loss += (torch.abs(wheelnumb - wheelpre)).mean() * 0.05
        a = []
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
            a.append(l * losses[k]["weight"])
        a.append(0)
        sum_loss += L1_loss(viewp,view0)
        sum_loss.backward()
        optimizer.step()

        print("[E: %d_%d] loss: %f, siloss: %f, edgeloss: %f, normloss: %f,laploss: %f,chamloss: %f, Dloss: %f"
              % (epoch,i,sum_loss.data, a[0].data, a[1].data, a[2].data, a[3].data, a[4].data,Dloss.data))

        if epoch % plot_period == 0:
            showresult.showresults(epoch, Bsize, new_src_mesh, tankpath, target_silhouette, target_rgb, temp_mesh)
            # model.SaveNet(saveroot, epoch)
        plt.close("all")
        plt.clf()
    if epoch % save_period == 0:
        model.SaveNet(saveroot,epoch)