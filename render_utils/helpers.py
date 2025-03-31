"""
    Various helper functions

    create_scene -> combines multiple nvdiffmodeling meshes in to a single mesh with mega texture
"""
import sys
import torch
import torch.nn.functional as F
from math import ceil

import nvdiffmodeling.src.mesh as mesh
import nvdiffmodeling.src.texture as texture
from nvdiffmodeling.src     import render
import imageio
import numpy as np

cosine_sim = torch.nn.CosineSimilarity()

def cosine_sum(features, targets):
    return -cosine_sim(features, targets).sum()


def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()


def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a * scale_a) + add_a, a_idx
    elif a is None and b is not None:
        return (b * scale_b) + add_b, b_idx
    else:
        return torch.cat(((a * scale_a) + add_a, (b * scale_b) + add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]),dim=0)

def to_image(tensor):
    if len(tensor.shape)==3:
        tensor = tensor.permute(1, 2, 0)
    elif len(tensor.shape)==4:
        tensor = tensor.permute(0, 2, 3, 1)
    arr = tensor.detach().cpu().numpy()
    #arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')

def unproject2to3(params, source_points_3d, target_points_2d,H,W):
    mvp = params['mvp'].squeeze()
    source_points_3d_clip = project3to2(params,source_points_3d)
    target_points_2d = target_points_2d/(torch.tensor([H, W],device=source_points_3d.device)//2)-1
    target_points_clip = torch.ones((source_points_3d_clip.shape[0],source_points_3d_clip.shape[1]+1),device=source_points_3d.device)
    target_points_clip[:,:2] = target_points_2d
    target_points_clip[:,2] = source_points_3d_clip[:,2]
    target_points_3d = torch.matmul( target_points_clip,torch.linalg.inv(mvp.T),
                               )  # [N, 4]
    target_points_3d = target_points_3d[:,:3] / target_points_3d[:,3:]
    return target_points_3d

def project3to2(params,points_3d, clip=True, H=1024, W=1024):
    mvp = params['mvp'][0]
    points_3d_clip = torch.matmul(F.pad(points_3d, (0, 1), 'constant', 1.0),
                                mvp.T)  # [N, 4]
    points_3d_clip[:, :3] /= points_3d_clip[:, 3:]  # perspective division
    if clip:
        return points_3d_clip[:,:3]
    else:
        points_2d = (((points_3d_clip[:, :2] + 1) / 2) * torch.tensor([H, W],device=points_3d.device)).round().type(torch.int64)
        return points_2d

class user_constraint_3d_dr:

    @torch.no_grad()
    def __init__(self,m ,glctx, params, cfg):
        self.params = params
        self.m = m.eval(params)
        self.cfg = cfg
        self.bkg = params['bkgs']
        self.glctx = glctx
        self.source_points_3d = []
        self.verts_idx = []
        self.t_idx = []
        self.uv = []
        self.device = self.m.v_pos.device
        img, rast = render.render_mesh(self.glctx,
                                       self.m,
                                       params['mvp'],
                                       params['campos'],
                                       params['lightpos'],
                                       self.cfg['light_power'],
                                       self.cfg['train_res'],
                                       spp=1,
                                       num_layers=1,
                                       msaa=False,
                                       background=self.bkg,
                                       return_rast=True)
        self.rast = rast

    def reset(self,source_points_2d,target_points_2d):
        self.source_points_3d = []
        self.verts_idx = []
        self.select_verts(source_points_2d)
        self.get_target_3d(target_points_2d)

    @torch.no_grad()
    def select_verts(self, source_points_2d):

        for i in range(len(source_points_2d)):
            rast = self.rast[0, int(source_points_2d[i,1]), int(source_points_2d[i,0])]
            # not hitting the mesh
            if rast[3] <= 0:
                return
            self.t_idx.append(rast[3].long() - 1)
            trig = self.m.t_pos_idx[rast[3].long() - 1]  # [3,]
            self.verts_idx.append(trig)
            vert = self.m.v_pos[trig.long()]  # [3, 3]
            uv = rast[:2]
            self.uv.append(uv)
            self.source_points_3d.append((1 - uv[0] - uv[1]) * vert[0] + uv[0] * vert[1] + uv[1] * vert[2])

    @torch.no_grad()
    def get_target_3d(self,target_points_2d):
        source_points_3d = torch.stack(self.source_points_3d)
        self.target_points_3d = unproject2to3(self.params,source_points_3d,target_points_2d,self.cfg['train_res'],self.cfg['train_res'])
        #self.delta_3d = target_points_3d-source_points_3d
        #verts_idx = torch.stack(self.verts_idx)
        #self.target_verts = self.m.v_pos[verts_idx]+self.delta_3d.unsqueeze(1)

    def track_source_points(self, m1):
        m_eval = m1.eval(self.params)
        verts_idx = torch.stack(self.verts_idx)
        uv = torch.stack(self.uv)
        vert = m_eval.v_pos[verts_idx]
        source_points_3d = (1 - uv[:, 0:1] - uv[:, 1:2]) * vert[:, 0] + uv[:, 0:1] * vert[:, 1] + uv[:, 1:2] * vert[:, 2]
        return source_points_3d

    def compute_loss(self, m1):

        l2loss = torch.nn.MSELoss()
        source_points_3d = self.track_source_points(m1)
        return l2loss(source_points_3d,self.target_points_3d)

class user_constraint_3d:

    def __init__(self,handle_v_idx, handle_pos):
        self.handle_v_idx = handle_v_idx
        self.handle_pos = handle_pos


    def compute_loss(self, m1):

        l2loss = torch.nn.MSELoss()
        verts = m1.v_pos
        source_points_3d = verts[self.handle_v_idx]
        return l2loss(source_points_3d,self.handle_pos)

@torch.no_grad()
def draw_points(img, source_points_2d, color=[1,0,0]):
    H = img.shape[0]
    W = img.shape[1]
    radius = int((H + W) / 2 * 0.008)
    buffer_overlay = torch.zeros(img.shape).to(img.device)
    color = torch.tensor(color).to(img.device)
    for i in range(len(source_points_2d)):

        # draw source point
        if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < W - radius and source_points_2d[
            i, 1] >= radius and source_points_2d[i, 1] < H - radius:
            buffer_overlay[source_points_2d[i, 1] - radius:source_points_2d[i, 1] + radius,
            source_points_2d[i, 0] - radius:source_points_2d[i, 0] + radius] += color
    overlay_mask = torch.sum(buffer_overlay,dim=-1,keepdim=True) == 0
    return img*overlay_mask + buffer_overlay

def create_scene(meshes, sz=1024):
    # Need to comment and fix code

    scene = mesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes) + 1

    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz * ny)
    h = int(sz * nx)

    dev = meshes[0].v_tex.device

    kd_atlas = torch.ones((1, w, h, 4)).to(dev)
    ks_atlas = torch.zeros((1, w, h, 3)).to(dev)
    kn_atlas = torch.ones((1, w, h, 3)).to(dev)

    for i, m in enumerate(meshes):
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = int(i / ny)

        sc_x = 1. / nx
        sc_y = 1. / ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.,
            scale_b=torch.tensor([sc_x, sc_y]).to(dev),
            add_a=0.,
            add_b=torch.tensor([sc_x * pos_x, sc_y * pos_y]).to(dev)
        )

        kd_atlas[:, pos_y * sz:(pos_y * sz) + sz, pos_x * sz:(pos_x * sz) + sz, :m.material['kd'].data.shape[-1]] = \
        m.material['kd'].data
        ks_atlas[:, pos_y * sz:(pos_y * sz) + sz, pos_x * sz:(pos_x * sz) + sz, :m.material['ks'].data.shape[-1]] = \
        m.material['ks'].data
        kn_atlas[:, pos_y * sz:(pos_y * sz) + sz, pos_x * sz:(pos_x * sz) + sz, :m.material['normal'].data.shape[-1]] = \
        m.material['normal'].data

        scene = mesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene
        )

    scene = mesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': texture.Texture2D(
                kd_atlas
            ),
            'ks': texture.Texture2D(
                ks_atlas
            ),
            'normal': texture.Texture2D(
                kn_atlas
            ),
        },
        base=scene  # gets uvs etc from here
    )

    return scene


def angle2rotation(angle_xyz):

    alpha = angle_xyz[:,0:1]
    beta = angle_xyz[:,1:2]
    gamma = angle_xyz[:,2:]
    rotation = [torch.cat([torch.cos(beta)*torch.cos(gamma),
                 torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma),
                 torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma)+torch.sin(alpha)*torch.sin(gamma)],dim=1),
                torch.cat([torch.cos(beta)*torch.sin(gamma),
                 torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma),
                 torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma)-torch.sin(alpha)*torch.cos(gamma)],dim=1),
                torch.cat([-torch.sin(beta),
                 torch.sin(alpha)*torch.cos(beta),
                 torch.cos(alpha)*torch.cos(beta)],dim=1)]
    rotation = (torch.stack(rotation)).permute(1,0,2)
    return rotation


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = F.normalize(x_raw,dim=1)  # batch*3
    z = torch.cross(x, y_raw,dim=1)  # batch*3
    z = F.normalize(z)  # batch*3
    y = torch.cross(z, x,dim=1)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix.unsqueeze(0)


def estimate_r_per_triangle(f_idx, v_orig, v):
    with torch.no_grad():
        faces_orig = v_orig[f_idx]
        faces = v[f_idx]
        '''
        centroid_orig = faces_orig.mean(dim=1,keepdim=True)
        centroid_orig = torch.broadcast_to(centroid_orig,(centroid_orig.shape[0],3,3))
        centroid = faces.mean(dim=1,keepdim=True)
        centroid = torch.broadcast_to(centroid,(centroid.shape[0],3,3))
        faces = faces-centroid+centroid_orig
        '''
        covariance = torch.einsum('ijk,ijl->ikl', faces_orig, faces)
        U, _, Vh = torch.linalg.svd(covariance)
        V = Vh.transpose(1,2)
        d = V @ (U.transpose(1,2))
        det = torch.linalg.det(d)
        sign = torch.sign(det).unsqueeze(1)
        sign = torch.broadcast_to(sign, (sign.shape[0],3))
        V[:, :, 2] = sign * V[:, :, 2]
        r = V @ U.transpose(1,2)
        return r
'''
def arap_t(rotation, s, ff_adj):

    r_adj = rotation[ff_adj]
    s_adj = s[ff_adj]
    r_adj_loss = 0
    #stiff = torch.diag(stiff)
    for i in range(3):
        #loss += (stiff@(rotation-r_adj[:,i])).norm()/ff_adj.shape[0]
        r_adj_loss += torch.mean(torch.square(rotation-r_adj[:,i]),dim=1)
    r_adj_loss = (r_adj_loss).mean()

    s_adj_loss=0
    for i in range(3):
        s_adj_loss += torch.mean(torch.square(s- s_adj[:, i]), dim=1)
    s_adj_loss = (s_adj_loss).mean()
    return 1 * r_adj_loss + 1 * s_adj_loss
'''
def boundary_loss(boundary_v,boundary_f,orig_v,v,rotation,s):
    bv_loss = torch.nn.functional.mse_loss(v[boundary_v], orig_v[boundary_v])
    s_boundary = s[boundary_f]
    r_boundary = rotation[boundary_f]
    s_identity = torch.zeros((s_boundary.shape[0], 6)).to(s.device)
    s_identity[:, 0] = 1
    s_identity[:, 3] = 1
    s_identity[:, 5] = 1
    r_identity = torch.zeros((s_boundary.shape[0], 6)).to(s.device)
    r_identity[:, 0] = 1
    r_identity[:, 4] = 1
    bf_loss = torch.mean(torch.norm(s_boundary - s_identity,dim=1))+torch.mean(torch.norm(r_boundary - r_identity,dim=1))
    return bv_loss+bf_loss

def boundary_loss_t(boundary_v,boundary_f,orig_v,v,t):
    bv_loss = torch.nn.functional.mse_loss(v[boundary_v], orig_v[boundary_v])
    t_boundary = t[boundary_f]

    t_identity = torch.zeros((t_boundary.shape[0], 9)).to(t.device)
    t_identity[:, 0] = 1
    t_identity[:, 4] = 1
    t_identity[:, 8] = 1

    bf_loss = torch.mean(torch.norm(t_boundary - t_identity,dim=1))
    return bv_loss+bf_loss

def reg_t(t,ff_adj, area):
    t_adj = t[ff_adj]
    area_adj = area[ff_adj]
    mean_area = area.mean()
    t_adj_loss = 0
    for i in range(3):
        #loss += (stiff@(rotation-r_adj[:,i])).norm()/ff_adj.shape[0]
        w = (area+area_adj[:,i])/2
        t_adj_loss += w*torch.norm(t-t_adj[:,i],dim=1)
        #r_adj_loss += w*torch.mean(torch.square(rotation-r_adj[:,i]),dim=1)
    t_adj_loss = (0.6*t_adj_loss).mean()/mean_area
    return t_adj_loss

def arap_t(rotation, s, ff_adj, area, alpha = 0.6, beta = 0.12, deformation_type = 0, stiff_mask=None):
    alpha_w = torch.ones(s.shape[0]).to(s.device)
    beta_w = torch.ones(s.shape[0]).to(s.device)
    r_adj = rotation[ff_adj]
    s_adj = s[ff_adj]
    area_adj = area[ff_adj]
    s_adj_loss = 0
    r_adj_loss = 0
    s_deviation_loss = 0
    r_deviation_loss = 0
    mean_area = area.mean()



    if deformation_type == 0:

        alpha_w *= alpha
        beta_w *= beta
        if stiff_mask is not None:
            alpha_w[stiff_mask] = alpha
            beta_w[stiff_mask] = 100
            s_stiff = s[stiff_mask]
            area_stiff = area[stiff_mask]
            s_identity = torch.zeros((s_stiff.shape[0], 6)).to(s.device)
            s_identity[:, 0] = 1
            s_identity[:, 3] = 1
            s_identity[:, 5] = 1
            s_deviation_loss = area_stiff * torch.norm(s_stiff - s_identity, dim=1)
            s_deviation_loss = (100 * s_deviation_loss).mean() / mean_area
    elif deformation_type == 1:  # rotation only
        alpha_w *= 5
        beta_w *= 100
        s_identity = torch.zeros((s.shape[0], 6)).to(s.device)
        s_identity[:, 0] = 1
        s_identity[:, 3] = 1
        s_identity[:, 5] = 1
        s_deviation_loss = area * torch.norm(s - s_identity, dim=1)
        s_deviation_loss = (100 * s_deviation_loss).mean() / mean_area
    else:  # stretching only
        alpha_w *= 100
        beta_w *= 1
        if stiff_mask is not None:
            beta_w[stiff_mask] = 100
            s_stiff = s[stiff_mask]
            area_stiff = area[stiff_mask]
            s_identity = torch.zeros((s_stiff.shape[0], 6)).to(s.device)
            s_identity[:, 0] = 1
            s_identity[:, 3] = 1
            s_identity[:, 5] = 1
            s_deviation_loss = area_stiff * torch.norm(s_stiff - s_identity, dim=1)
            s_deviation_loss = (100 * s_deviation_loss).mean() / mean_area
        r_identity = torch.zeros((rotation.shape[0], 6)).to(s.device)
        r_identity[:, 0] = 1
        r_identity[:, 4] = 1
        r_deviation_loss = area * torch.norm(rotation - r_identity, dim=1)
        r_deviation_loss = (100 * r_deviation_loss).mean() / mean_area




    for i in range(3):
        #loss += (stiff@(rotation-r_adj[:,i])).norm()/ff_adj.shape[0]
        w = (area+area_adj[:,i])/2
        r_adj_loss += w*torch.norm(rotation-r_adj[:,i],dim=1)
        #r_adj_loss += w*torch.mean(torch.square(rotation-r_adj[:,i]),dim=1)
    r_adj_loss = (alpha_w*r_adj_loss).mean()/mean_area


    for i in range(3):
        w = (area + area_adj[:, i]) / 2
        s_adj_loss += w * torch.norm(s-s_adj[:,i],dim=1)
        #s_adj_loss += w*torch.mean(torch.square(s- s_adj[:, i]), dim=1)
    s_adj_loss = (beta_w*s_adj_loss).mean()/mean_area


    return r_adj_loss + s_adj_loss + s_deviation_loss + r_deviation_loss








def to_symmetric_s(jacobians):

    S = torch.stack([jacobians[:,0],jacobians[:,1],jacobians[:,2],
                      jacobians[:,1],jacobians[:,3],jacobians[:,4],
                      jacobians[:,2],jacobians[:,4],jacobians[:,5]])
    S = S.transpose(0,1)
    S = S.reshape((S.shape[0],3,3)).unsqueeze(0)
    return S
'''
def to_symmetric_s(jacobians,gt_jacobians):

    S = torch.stack([gt_jacobians[:,0,0],jacobians[:,0],jacobians[:,1],
                      jacobians[:,0],gt_jacobians[:,1,1],jacobians[:,2],
                      jacobians[:,1],jacobians[:,2],gt_jacobians[:,2,2]])
    S = S.transpose(0,1)
    S = S.reshape((S.shape[0],3,3)).unsqueeze(0)
    return S
'''
def mul_diag(r,s):
    for i in range(3):
        r[:,:,i,i] *= s[:,i]
    return r


"""
    Helper class to create and add images to video
"""



class Video():
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:

        if path[-1] != "/":
            path += "/"

        self.writer = imageio.get_writer(path + name, mode=mode, fps=fps, codec=codec, bitrate=bitrate)

    def ready_image(self, image, write_video=True):
        # assuming channels last - as renderer returns it
        if len(image.shape) == 4:
            image = image.squeeze(0)[..., :3].detach().cpu().numpy()
        else:
            image = image[..., :3].detach().cpu().numpy()

        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

        if write_video:
            self.writer.append_data(image)

        return image

    def close(self):
        self.writer.close()


"""
    Helper function to read anchor and handle
"""
def read_anchor_and_handle(path, size ,device):
    anchors_mask_f = torch.zeros(size,dtype=torch.bool)
    stiff_mask_f = torch.zeros(size, dtype=torch.bool)
    handles_idx_v = []
    handles_pos = []
    with open(path,'r') as f:
        for line in f.readlines():
            if line=="#anchor_f_index\n":
                mode = 0
                continue
            elif line=="#handle_v_index\n":
                mode = 1
                continue
            elif line=="#stiff_f_index\n":
                mode = 2
                continue
            elif line=="#handle_target_position\n":
                mode = 3
                continue
            if (mode==0):
                idx = int(line)
                anchors_mask_f[idx] = True
            elif (mode==1):
                idx = int(line)
                handles_idx_v.append(idx)
            elif (mode==2):
                idx = int(line)
                stiff_mask_f[idx] = True
            elif (mode==3):
                p = line.split(',')
                handles_pos.append([float(p[0]),float(p[1]),float(p[2])])

    handles_idx_v = np.array(handles_idx_v)
    handles_pos = np.array(handles_pos)
    handles_idx_v = torch.from_numpy(handles_idx_v)
    handles_pos = (torch.from_numpy(handles_pos)).to(torch.float32)
    return anchors_mask_f.to(device),stiff_mask_f.to(device), handles_idx_v.to(device),handles_pos.to(device)



