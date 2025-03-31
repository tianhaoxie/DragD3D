
import imageio.v2
import os

import numpy

from nvdiffmodeling.src import texture, render, mesh, obj, regularizer
from render_utils.render import load_mesh, create_cams, rgb_to_srgb, srgb_to_rgb
from render_utils.helpers import user_constraint_3d_dr, draw_points, project3to2, Video, to_image, compute_rotation_matrix_from_ortho6d,arap_t,to_symmetric_s,boundary_loss,boundary_loss_t,reg_t
from render_utils.camera import get_camera_params

from render_utils.resize_right import resize
import nvdiffrast.torch as dr
import torch
from NeuralJacobianFields import SourceMesh
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import yaml
import numpy as np
from diffusion_prior.sds_utils import sds_loss
from Adan import Adan

import igl
import wandb
from arap_deformer import arap_deformer
import  time as t

class ARAP_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, arap_obj):
        arap_obj.compute_loss(v)
        energy = arap_obj.get_energy()
        ctx.in1 = arap_obj.get_J()
        return torch.tensor([energy], device='cuda:0')

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.in1
        return grad * grad_output, None, None


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_ref(path, device):
    num_ref = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    refs = []
    for i in range(num_ref):
        img = Image.open(os.path.join(path, '{}.png'.format(i)))
        t = transforms.ToTensor()
        img = t(img).to(device).squeeze()
        refs.append(img)
    return refs


def get_boundary_f(f):
    boundary_edge = igl.boundary_facets(f.detach().cpu().numpy())
    boundary_v = np.unique(boundary_edge.flatten())
    boundary_v_t = torch.from_numpy(boundary_v).to(f.device)
    boundary_f = torch.zeros(f.shape[0],dtype=torch.bool,device=f.device)
    for i in range(len(boundary_v_t)):
        bf_mask = (f == boundary_v_t[i])
        bf_mask = torch.bitwise_or(torch.bitwise_or(bf_mask[:,0] , bf_mask[:,1]),bf_mask[:,2])
        boundary_f = torch.bitwise_or(bf_mask, boundary_f)
    boundary_f = torch.nonzero(boundary_f).squeeze()
    return boundary_f

def get_mesh_from_f_mask(f_mask, v, f):
    with torch.no_grad():
        masked_f_idx = f_mask.nonzero().squeeze(1)
        masked_v_idx = torch.flatten(f[masked_f_idx], 1).unique()
        new_v = v[masked_v_idx]
        orig_v_idx = torch.arange(v.shape[0], device=v.device)
        new_v_idx = torch.arange(new_v.shape[0], device=v.device)
        orig2new = torch.scatter(orig_v_idx, dim=0, index=masked_v_idx, src=new_v_idx)
        new_f = orig2new[f]
        new_f = new_f[masked_f_idx]
    boundary_edge = igl.boundary_facets(new_f.detach().cpu().numpy())
    boundary_v = np.unique(boundary_edge.flatten())


    return new_v, new_f, boundary_v, masked_v_idx

def get_stiff_from_mask(f_mask,stiff_mask):
    with torch.no_grad():
        masked_f_idx = f_mask.nonzero().squeeze(1)
        stiff = stiff_mask[masked_f_idx]
    return stiff

def get_value_of_mask_v(v, mask_v, value):
    mask_v = mask_v[:, None].expand(mask_v.shape[0], 3)
    v = torch.scatter(v, dim=0, index=mask_v, src=value)
    return v

def get_rotation_of_mask_stiff(rotation, mask_stiff, value):
    mask_stiff = mask_stiff[:, None].expand(mask_stiff.shape[0], 6)
    stiff = torch.scatter(rotation, dim=0, index=mask_stiff, src=value)
    return stiff

def c_weight_scheduler(epoch, total_epoch, c_weight, type='linear'):
    if type == 'nonlinear':
        c_loss_weight = 1 + (c_weight - 1) * (epoch / total_epoch) ** 2
    elif type == 'linear':
        c_loss_weight = 1 + (c_weight - 1) * (epoch / total_epoch)
    elif type == 'fixed':
        c_loss_weight = c_weight
    else:
        raise Exception("wrong type")
    return c_loss_weight

def train(cfg, model, experiment, cam_dis, drag_azim,alpha ,beta, deformation_type, guidance_type, njf=True, using_r_loss = True, random_camera = True, using_texture = True, using_t_matrix = False):
    seed_everything(666)
    '''
    wandb.init(
        # set the wandb project where this run will be logged
        project="drag3d",

        # track hyperparameters and run metadata
        config={
            "learning_rate": cfg['lr'],
            "model": model,
            "experiment": experiment,
            "guidance_type": guidance_type,
            "epochs": cfg['epochs'],
            "alpha": cfg['r_loss_alpha']

        }
    )
    '''
    glctx = dr.RasterizeGLContext()

    m_path = './shapes/{}/mesh/mesh.obj'.format(model)

    m = load_mesh(m_path)
    m.material['bsdf'] = cfg['bsdf']
    # bkg = read_bkg('./horse/bkg.jpeg', cfg['train_res']).permute(1, 2, 0).unsqueeze(0).cuda()
    # bkg = srgb_to_rgb(bkg)

    # four standard views
    bkg = torch.ones((1, cfg['train_res'], cfg['train_res'], 3)).cuda()
    params_side = get_camera_params(cfg['elev_angle'], drag_azim % 360, cam_dis, cfg["train_res"], bkg)
    params_front = get_camera_params(cfg['elev_angle'], (drag_azim + 90) % 360, cam_dis, cfg["train_res"], bkg)
    params_op_side = get_camera_params(cfg['elev_angle'], (drag_azim + 180) % 360, cam_dis, cfg["train_res"], bkg)
    params_back = get_camera_params(cfg['elev_angle'], (drag_azim + 270) % 360, cam_dis, cfg["train_res"], bkg)
    params_log = {}
    for key in params_side:
        param_side = params_side[key]
        param_front = params_front[key]
        param_op_side = params_op_side[key]
        param_back = params_back[key]
        param = torch.cat([param_side, param_front, param_op_side, param_back], dim=0)
        params_log[key] = param

    vertices = m.v_pos.clone().detach().requires_grad_(True)
    faces = m.t_pos_idx.clone().detach()
    '''
    kd_ = m.material['kd'].data.permute(0, 3, 1, 2)
    ks_ = m.material['ks'].data.permute(0, 3, 1, 2)

    try:
        nrml_ = m.material['normal'].data.permute(0, 3, 1, 2)
    except:
        nrml_ = torch.zeros(kd_.shape, device=kd_.device)
        #nrml_[:, 2:] = 1
    # convert all texture maps to trainable tensors
    texture_map = texture.create_trainable(
        resize(kd_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1),
        [cfg["texture_resolution"]] * 2, True)
    specular_map = texture.create_trainable(
        resize(ks_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1),
        [cfg["texture_resolution"]] * 2, True)
    normal_map = texture.create_trainable(
        resize(nrml_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1),
        [cfg["texture_resolution"]] * 2, True)
    '''
    m = mesh.Mesh(
        vertices,
        faces,

        # material={
        #   'bsdf': cfg['bsdf'],
        #  'kd': texture_map,
        # 'ks': specular_map,
        # 'normal': normal_map
        # },
        base=m
    )
    if using_texture == False:
        with torch.no_grad():
            gray = torch.zeros((1, 3, cfg["texture_resolution"], cfg["texture_resolution"])).cuda()
            gray = torch.fill(gray, 0.3)
            texture_map = texture.create_trainable(
                resize(gray, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1),
                [cfg["texture_resolution"]] * 2, True)
            m.material['kd'] = texture_map
            m.material['bsdf'] = 'diffuse'

    handles = torch.load('./shapes/{}/{}/points.pt'.format(model, experiment), map_location='cuda:0')
    source_points2d = torch.stack(handles['handle'], dim=0) * (cfg['train_res'] // 512)
    target_points2d = torch.stack(handles['target'], dim=0) * (cfg['train_res'] // 512)
    #target_points2d = source_points2d+(1.5*(target_points2d-source_points2d)).to(torch.int)

    m_ready = mesh.auto_normals(m)
    m_ready = mesh.compute_tangents(m_ready)
    uc3d = user_constraint_3d_dr(m_ready, glctx, params_side, cfg)
    uc3d.reset(source_points2d, target_points2d)

    if guidance_type is not None:

        if using_r_loss:
            log_path = './shapes/{}/{}/{}_arap'.format(model, experiment, guidance_type)
        elif using_t_matrix:
            log_path = './shapes/{}/{}/{}_t'.format(model, experiment, guidance_type)
        else:
            log_path = './shapes/{}/{}/{}'.format(model, experiment, guidance_type)
    else:
        log_path = './shapes/{}/{}/j_arap'.format(model, experiment)

    if njf == False:
        log_path = log_path + '_wonjf'
    elif using_texture == False:
        log_path = log_path + '_wotex'
    elif random_camera == False:
        log_path = log_path + '_fixcam'

    if os.path.isdir(log_path) == False:
        os.mkdir(log_path)
    log_video = Video(log_path)

    with torch.no_grad():
        undeformed_img = render.render_mesh(glctx,
                                            m_ready.eval(params_log),
                                            params_log['mvp'],
                                            params_log['campos'],
                                            params_log['lightpos'],
                                            cfg['light_power'],
                                            cfg['train_res'],
                                            spp=1,
                                            num_layers=cfg["layers"],
                                            msaa=False,
                                            background=bkg)

        undeformed_img = rgb_to_srgb(undeformed_img).permute(0, 3, 1, 2)[0]
        orig = undeformed_img.detach().clone().permute(1, 2, 0)
        orig = draw_points(orig, source_points2d)
        #target_points_3d = uc3d.target_points_3d
        #target_points_2d = project3to2(params_log, target_points_3d, False, cfg['train_res'], cfg['train_res'])
        orig = draw_points(orig, target_points2d, [0, 0, 1])
        imageio.imsave(os.path.join(log_path, 'start.png'),
                       to_image(orig.permute(2, 0, 1)))

    # bkg = torch.ones((1, cfg['train_res'], cfg['train_res'], 3), device='cuda:0')
    # ref_imgs = load_ref('./{}/{}/ref/'.format(model,experiment),'cuda:0')
    try:
        f_mask = torch.load('./shapes/{}/{}/{}_f_mask.pt'.format(model, experiment, experiment))
        #stiff_mask = torch.load('./shapes/{}/{}/stiff_mask.pt'.format(model, experiment))

        #masked_f_idx = f_mask.nonzero().squeeze(1)
        nv, nf, boundary_v, v_mask = get_mesh_from_f_mask(f_mask, m.v_pos, m.t_pos_idx)
        #stiff_mask = get_stiff_from_mask(f_mask, stiff_mask)
        #stiff = (stiff_mask) * (10 - 1) + torch.ones(stiff_mask.shape, device=stiff_mask.device)

        # if produce more than one component by masking, select component by user constraint
        fc = igl.facet_components(nf.detach().cpu().numpy())
        if (fc.max() != 0):
            fc = torch.from_numpy(fc).cuda()
            tf_idx = uc3d.t_idx[0]
            tnf_idx = torch.where(f_mask.nonzero() == tf_idx)[0]
            tc = fc[tnf_idx]
            fc_mask = torch.where(fc == tc, True, False)
            f_mask_n = torch.zeros(f_mask.shape, dtype=bool).cuda()
            f_mask_n[f_mask.nonzero().squeeze()] = fc_mask
            masked_f_idx = f_mask_n.nonzero().squeeze(1)
            nv, nf, boundary_v, v_mask = get_mesh_from_f_mask(f_mask_n, m.v_pos, m.t_pos_idx)


        bv = torch.from_numpy(boundary_v).cuda()
        bv_mask_n = torch.zeros(nv.shape[0], 1).cuda()
        bv_mask_n[bv] = 1
        bv_mask = torch.zeros(m.v_pos.shape[0], 1).cuda()
        bv_mask = torch.fill(bv_mask, 0.3)
        bv_mask = torch.scatter(bv_mask, dim=0, index=v_mask.unsqueeze(1), src=bv_mask_n)

        flag_mask = True

    except:
        nv = m.v_pos.detach().clone()
        nf = m.t_pos_idx.clone()
        boundary_v = np.array([])
        bv_mask = torch.ones(nv.shape[0], 1).cuda()
        flag_mask = False

    try:
        #stiff_mask = torch.load('./shapes/{}/{}/stiff_mask.pt'.format(model, experiment))
        stiff_idx = np.loadtxt('./shapes/{}/{}/stiff_mask.txt'.format(model, experiment),dtype=int)
        stiff_idx = torch.from_numpy(stiff_idx)
        stiff_mask = torch.zeros(faces.shape[0], dtype = torch.int)
        stiff_mask[stiff_idx] = 1
        stiff_mask = get_stiff_from_mask(f_mask, stiff_mask.cuda())
        stiff = torch.nonzero(stiff_mask).squeeze()

        #stiff = (stiff_mask) * (50 - 1) + torch.ones(stiff_mask.shape, device=stiff_mask.device)
        #stiff = 1/stiff
        print('using stiffness')
    except:

        stiff = None


    if njf == False:
        nv.requires_grad_(True)

    # v_handle = torch.load('./{}/{}/{}_handle.pt'.format(model,experiment,experiment),map_location='cuda:0')
    # delta_v_handle = v_handle['delta'].float()
    # handle_verts_idx = v_handle['vert_idx']
    # target_verts = m.v_pos[handle_verts_idx]+delta_v_handle.unsqueeze(1)
    # igl.write_obj('./mask_mesh.obj',nv.detach().cpu().numpy(),nf.cpu().numpy())

    # f = m.t_pos_idx.detach().cpu().numpy()
    f = nf.detach().cpu().numpy()

    jacobian_source = SourceMesh.SourceMesh(0, '', {}, 1, ttype=torch.float,
                                            mask=boundary_v
                                            )

    # Training parameters

    # train_params += texture_map.getMips()
    # train_params += normal_map.getMips()
    # train_params += specular_map.getMips()

    cams = create_cams(cfg)
    if guidance_type is not None:
        sds_l = sds_loss(opt=cfg, guidance=guidance_type, device='cuda:0')
    #alpha = cfg['r_loss_alpha']
    #beta = cfg['r_loss_beta']

    v = nv.detach().cpu().numpy()
    jacobian_source.load(v, f)
    igl.write_obj('./shapes/{}/{}/mask_mesh.obj'.format(model, experiment), v, f)
    jacobian_source.to('cuda:0')
    darea = igl.doublearea(v,f)
    darea = torch.from_numpy(darea).cuda()
    ff_adj, _ = igl.triangle_triangle_adjacency(f)
    ff_adj = torch.from_numpy(ff_adj).cuda()
    # in case of having boundary
    index = torch.broadcast_to(torch.arange(ff_adj.shape[0]).unsqueeze(1), ff_adj.shape).cuda()
    ff_adj = torch.where(ff_adj == -1, index, ff_adj)
    boundary_f = get_boundary_f(nf)


    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(nv.unsqueeze(0))

    #s = torch.stack([gt_jacobians[:, 0, 1], gt_jacobians[:, 0, 2], gt_jacobians[:, 1, 2]])

    #gt_jacobians.requires_grad_(True)

    t = torch.zeros((gt_jacobians.shape[1], 9)).cuda()
    t[:, 0] = 1
    t[:, 4] = 1
    t[:, 8] = 1
    t.requires_grad_(True)

    s = torch.zeros((gt_jacobians.shape[1],6)).cuda()
    s[:,0] = 1
    s[:,3] = 1
    s[:,5] = 1
    s.requires_grad_(True)

    #boundary_f = get_boundary_f(nv, nf)
    ortho6d = torch.zeros(nf.shape[0], 6).cuda()
    ortho6d[:, 0] = 1
    ortho6d[:, 4] = 1
    ortho6d.requires_grad_(True)

    pbar = tqdm(range(cfg['epochs']))


    train_params = []
    if njf:
        if using_t_matrix:
            train_params += [t]
        else:
            train_params += [ortho6d]
            train_params += [s]
        #train_params += [stiff]
    else:
        train_params += [nv]
    # train_params += normal_map.getMips()
    # Optimizer and Scheduler
    # optimizer = torch.optim.Adam(train_params, lr=cfg["lr"])
    optimizer = Adan(train_params, lr=cfg["lr"])
    #optimizer_stiff = Adan([stiff], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10 ** (-x * 0.0002)))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    for i in pbar:
        if random_camera:
            cams_params_train = next(iter(cams))
            for key in cams_params_train:
                cams_params_train[key] = cams_params_train[key].cuda()
            azim = cams_params_train['azim']
        else:
            cams_params_train = params_log
            azim = torch.tensor([[90], [180], [-90], [0]]).cuda()

        with torch.no_grad():
            undeformed_img = render.render_mesh(glctx,
                                                m_ready.eval(cams_params_train),
                                                cams_params_train['mvp'],
                                                cams_params_train['campos'],
                                                cams_params_train['lightpos'],
                                                cfg['light_power'],
                                                cfg['train_res'],
                                                spp=1,
                                                num_layers=cfg["layers"],
                                                msaa=False,
                                                background=cams_params_train['bkgs'])

            undeformed_img = rgb_to_srgb(undeformed_img).permute(0, 3, 1, 2)

        if njf:
            if using_t_matrix:
                trans = t.reshape(-1,3,3)
            else:
                rotation = compute_rotation_matrix_from_ortho6d(ortho6d)
                full_s = to_symmetric_s(s)
                trans = rotation @ full_s

            n_verts = jacobian_source.vertices_from_jacobians(trans@gt_jacobians).squeeze()
            if using_t_matrix:
                b_loss = boundary_loss_t(boundary_v,boundary_f,nv,n_verts,t)
            else:
                b_loss = boundary_loss(boundary_v, boundary_f, nv, n_verts, ortho6d, s)

            if (flag_mask):
                n_verts = get_value_of_mask_v(m.v_pos, v_mask, n_verts)
                #ortho6d_full = get_rotation_of_mask_stiff(ortho6d_i, masked_f_idx, ortho6d)

        else:
            n_verts = get_value_of_mask_v(m.v_pos, v_mask, nv)

        m1 = mesh.Mesh(
            n_verts,
            m.t_pos_idx,
            # material={
            #    'bsdf': cfg['bsdf'],
            #    'kd': texture_map,
            #    'ks': specular_map,
            #    #'normal': normal_map,
            # },
            base=m  # gets uvs etc from here
        )

        # m1 = m
        m1_ready = mesh.auto_normals(m1)
        m1_ready = mesh.compute_tangents(m1_ready)
        img = render.render_mesh(glctx,
                                 m1_ready.eval(cams_params_train),
                                 cams_params_train['mvp'],
                                 cams_params_train['campos'],
                                 cams_params_train['lightpos'],
                                 cfg['light_power'],
                                 cfg['train_res'],
                                 spp=1,
                                 num_layers=cfg["layers"],
                                 msaa=False,
                                 background=cams_params_train['bkgs'])

        img = rgb_to_srgb(img).permute(0, 3, 1, 2)
        # img = transforms.functional.resize(img, ref_imgs[step].shape[-1])

        c_loss = uc3d.compute_loss(m1_ready)
        c_loss_weight = c_weight_scheduler(i, cfg['epochs'], cfg['c_loss_weight'], 'fixed')
        # j_loss = ((gt_jacobians-orig_jacobians)**2).mean()


        r_loss = arap_t(ortho6d, s, ff_adj, darea, alpha=alpha, beta=beta, deformation_type=deformation_type,
                        stiff_mask=stiff)
        t_loss = reg_t(t,ff_adj,darea)
        if guidance_type is not None:
            with torch.autocast("cuda"):
                s_loss = sds_l.compute(azim, img, undeformed_img)
            if using_r_loss:
                loss = c_loss_weight * c_loss + r_loss + 100 * b_loss + s_loss
                # loss = s_loss+r_loss
            elif using_t_matrix:
                loss = c_loss_weight * c_loss + t_loss + 100 * b_loss + s_loss
            else:
                loss = c_loss_weight * c_loss + s_loss + 100 * b_loss
        else:
            loss = c_loss_weight * c_loss + r_loss + 100 * b_loss



        optimizer.zero_grad()
        #optimizer_stiff.zero_grad()
        loss.backward()
        optimizer.step()
        #optimizer_stiff.step()

        scheduler.step()
        # logging
        if (i % 10 == 0):
            with torch.no_grad():
                img_log = render.render_mesh(glctx,
                                             m1_ready.eval(params_log),
                                             params_log['mvp'],
                                             params_log['campos'],
                                             params_log['lightpos'],
                                             cfg['light_power'],
                                             cfg['train_res'],
                                             spp=1,
                                             num_layers=cfg["layers"],
                                             msaa=False,
                                             background=bkg)

                img_log = rgb_to_srgb(img_log).permute(0, 3, 1, 2)[0]
                img_log = draw_points(img_log.permute(1, 2, 0), target_points2d, [0, 0, 1])
                source_points_3d = uc3d.track_source_points(m1_ready)
                source_points_2d = project3to2(params_log, source_points_3d, False, cfg['train_res'], cfg['train_res'])
                img_log = draw_points(img_log, source_points_2d, [1, 0, 0])
                img_log = log_video.ready_image(img_log)
        # normal_map.clamp_(min=-1, max=1)
        # specular_map.clamp_(min=0, max=1)
        # texture_map.clamp_(min=0, max=1)
        # writer.add_scalar('image loss', loss.item()-r_loss_weight * r_loss.item()-j_loss_weight*j_loss, i)
        # writer.add_scalar('arap loss', r_loss.item(), i)
        # writer.add_scalar('j loss', j_loss.item(), i)
        # pbar.set_description('image loss = {}, arap loss = {}, j loss = {} '.format(loss.item()-r_loss_weight * r_loss.item()-j_loss_weight*j_loss, r_loss.item(), j_loss.item()))
        pbar.set_description('c loss = {}, r loss = {}, b_loss = {} '.format(
            c_loss.item(), r_loss.item()/alpha, b_loss.item()))
        #if logging_wandb:

        #wandb.log({"c_loss": c_loss.item(), "r_loss":r_loss.item()})

    log_video.close()
    with torch.no_grad():

        img = render.render_mesh(glctx,
                                 m1_ready.eval(params_log),
                                 params_log['mvp'],
                                 params_log['campos'],
                                 params_log['lightpos'],
                                 cfg['light_power'],
                                 cfg['train_res'],
                                 spp=1,
                                 num_layers=cfg["layers"],
                                 msaa=False,
                                 background=bkg)

        img = rgb_to_srgb(img).permute(0, 3, 1, 2)
        '''
        img_side = img[0]
        img_ = draw_points(img_side.permute(1,2,0), target_points_2d , [0, 0, 1])
        source_points_3d = uc3d.track_source_points(m1_ready)
        source_points_2d = project3to2(params_log,source_points_3d,False,cfg['train_res'],cfg['train_res'])
        img_ = draw_points(img_, source_points_2d, [1, 0, 0])
        '''
        # imageio.imsave(os.path.join(log_path, 'optimized.png'), to_image(img_.permute(2,0,1)))
        # imageio.imsave(os.path.join(log_path, 'optimized.png'), to_image(img_side))
        img_log = img[0]
        for i in range(3):
            img_log = torch.cat([img_log, img[i + 1]], dim=2)
        imageio.imsave(os.path.join(log_path, 'optimized.png'), to_image(img_log))

    obj.write_obj(log_path, m1)
    #torch.save(s,log_path+'_s.pt')



if __name__ == '__main__':

    models = {'name': ['chair'],
              'experiment': 'exp1',
              'guidance_type': ['SD'],
              'deformation_type': 0,
              'njf': True,
              'alpha': 0.6,
              'beta': [0.12]
              }

    # model = 'chair'
    # experiment = 'exp1'
    # guidance_type = 'SD'
    for model in models['name']:
        for guidance_type in models['guidance_type']:

            with open('./configs/config_{}.yml'.format(model), 'r') as stream:
                try:
                    cfg = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)


            for beta in models['beta']:

                print('Experiment of model {}, {}, {} guidance, njf {}'.format(model, models['experiment'], guidance_type, models['njf']))
                train(cfg=cfg,
                      model=model,
                      experiment=models['experiment'],
                      cam_dis=cfg['cam_dis'],
                      drag_azim=cfg['drag_azim'],
                      alpha=models['alpha'],
                      beta=beta,
                      deformation_type=models['deformation_type'],
                      guidance_type=guidance_type,
                      njf=models['njf'],
                      using_r_loss=True,
                      using_texture=True,
                      random_camera=True,
                      using_t_matrix=False)