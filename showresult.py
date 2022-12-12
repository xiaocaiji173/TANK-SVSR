import torch
import os

from train_utils import *


def showresults(epoch,Bsize,new_src_mesh,tankpath,target_silhouette,target_rgb,temp_mesh):
    if not os.path.exists(saveroot + '/resultsimg/epoch_' + str(epoch)):
        os.makedirs(saveroot + '/resultsimg/epoch_' + str(epoch))
    if not os.path.exists(saveroot + '/results/epoch_' + str(epoch)):
        os.makedirs(saveroot + '/results/epoch_' + str(epoch))
    if not os.path.exists(saveroot + '/resultstemp/epoch_' + str(epoch)):
        os.makedirs(saveroot + '/resultstemp/epoch_' + str(epoch))
    if not os.path.exists(saveroot + '/resultsimgtemp/epoch_' + str(epoch)):
        os.makedirs(saveroot + '/resultsimgtemp/epoch_' + str(epoch))

    tankpath = ''.join(tankpath)
    for i in range(Bsize):
        final_img = os.path.join(saveroot + '/resultsimg/epoch_' + str(epoch),
                                 str(epoch)  + '_' + tankpath + '_img.png')
        visualize_prediction(new_src_mesh[i], renderer=renderer_silhouette, title="iter: %d" % i, silhouette=True,
                             target_image=target_silhouette[3][i, :, :], filepath=final_img)
        final_img = os.path.join(saveroot + '/resultsimg/epoch_' + str(epoch),
                                 str(epoch)+ '_' + tankpath + '_img2.png')
        visualize_prediction(new_src_mesh[i], renderer=renderer_textured, title="iter: %d" % i, silhouette=False,
                             target_image=target_rgb[3][i, :, :, :], filepath=final_img)

        final_img = os.path.join(saveroot + '/resultsimgtemp/epoch_' + str(epoch),
                                 str(epoch) + '_' + tankpath + '_img.png')
        visualize_prediction(temp_mesh, renderer=renderer_silhouette, title="iter: %d" % i, silhouette=True,
                             target_image=target_silhouette[3][i, :, :], filepath=final_img)

        new_src_mesh = new_src_mesh.scale_verts(1 / vertexscale)
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(i)
        final_obj = os.path.join(saveroot + '/results/epoch_' + str(epoch),
                                 str(epoch)  + '_' + tankpath + 'final_model.obj')
        save_obj(final_obj, final_verts, final_faces)
        final_verts, final_faces = temp_mesh.get_mesh_verts_faces(i)
        final_obj = os.path.join(saveroot + '/resultstemp/epoch_' + str(epoch),
                                 str(epoch)+ '_' + tankpath + 'temp_model.obj')
        save_obj(final_obj, final_verts, final_faces)