
import imageio
from render_utils.camera           import CameraBatch, get_camera_params
from render_utils.resize_right import resize
from nvdiffmodeling.src     import obj,texture,mesh,render
import igl
import nvdiffrast.torch     as dr
import yaml
import torch

import os
from render_utils.helpers import user_constraint_3d,draw_points,project3to2,to_image


def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def load_mesh(path, unit=True, return_scale = False):
    try:
        loaded_mesh = obj.load_obj(path)
    except:
        import pymeshlab
        import numpy as np
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        if not ms.current_mesh().has_wedge_tex_coord():
            # some arbitrarily high number
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
        ms.save_current_mesh('./tmp/tmp.obj')
        loaded_mesh = obj.load_obj('./tmp/tmp.obj')
        texture_map = texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda'))
        specular_map = texture.Texture2D(torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda'))
        loaded_mesh = mesh.Mesh(
        material={
            'bsdf': 'pbr',
            'kd': texture_map,
            'ks': specular_map,
        },
        base=loaded_mesh # Get UVs from original loaded mesh
        )
    if unit:
        if return_scale:
            loaded_mesh,scale = mesh.unit_size(loaded_mesh,True)
            return loaded_mesh, scale
        else:
            loaded_mesh = mesh.unit_size(loaded_mesh)

    return loaded_mesh

def preprocessing_mesh(path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.compute_normal_per_vertex()
    ms.save_current_mesh(path)


def create_cams(cfg,center=[0,0,0]):
    cams_data = CameraBatch(
        cfg["train_res"],
        [cfg["dist_min"], cfg["dist_max"]],
        [cfg["azim_min"], cfg["azim_max"]],
        [cfg["elev_alpha"], cfg["elev_beta"], cfg["elev_max"]],
        [cfg["fov_min"], cfg["fov_max"]],
        cfg["aug_loc"],
        cfg["aug_light"],
        cfg["aug_bkg"],
        cfg["batch_size"],
        look_at=center
    )
    cams = torch.utils.data.DataLoader(
        cams_data,
        cfg["batch_size"],
        num_workers=0,
        pin_memory=True
    )
    return cams



if __name__=='__main__':

    '''
    models = ['nascar_remesh',
              'cat',
              'castle',
              'chair',
              'elephant',
              'rocket',
              'snow_monster']
    '''
    models = ['rocket']
    for model in models:
        with open('../configs/config_{}.yml'.format(model), 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        glctx = dr.RasterizeGLContext()

        cam_dis = cfg['cam_dis']
        experiment = 'exp2'
        #elev_angle = cfg['elev_angle']
        elev_angle = 0
        #m = load_mesh('../shapes/{}/mesh/mesh.obj'.format(model))
        m = load_mesh('../shapes/{}/{}/DDS_arap/mesh.obj'.format(model,experiment),False)

        m.material['bsdf'] = cfg['bsdf']
        res = 1024
        cfg['train_res'] = res
        '''
        m.material['bsdf'] = 'diffuse'
        with torch.no_grad():
            gray = torch.zeros((1,3,cfg["texture_resolution"],cfg["texture_resolution"])).cuda()
            gray = torch.fill(gray,0.3)
            texture_map = texture.create_trainable(
                resize(gray, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1),
                [cfg["texture_resolution"]] * 2, True)
            m.material['kd'] = texture_map
        '''
        bkg = torch.ones((1, res, res, 3)).cuda()
        params_side = get_camera_params(elev_angle, cfg['drag_azim'] % 360, cam_dis, res, bkg)
        params_front = get_camera_params(elev_angle, (cfg['drag_azim'] + 90) % 360, cam_dis, res, bkg)
        params_op_side = get_camera_params(elev_angle, (cfg['drag_azim'] + 180) % 360, cam_dis, res, bkg)
        params_back = get_camera_params(elev_angle, (cfg['drag_azim'] + 270) % 360, cam_dis, res, bkg)
        params = {}
        for key in params_side:
            param_side = params_side[key]
            param_front = params_front[key]
            param_op_side = params_op_side[key]
            param_back = params_back[key]
            param = torch.cat([param_side, param_front, param_op_side, param_back], dim=0)
            params[key] = param

        m = mesh.auto_normals(m)
        m = mesh.compute_tangents(m)
        img = render.render_mesh(glctx,
                                 m.eval(params),
                                 params['mvp'],
                                 params['campos'],
                                 params['lightpos'],
                                 cfg['light_power'],
                                 res,
                                 spp=1,
                                 num_layers=cfg["layers"],
                                 msaa=False,
                                 background=bkg)
        img = rgb_to_srgb(img).permute(0, 3, 1, 2)
        rendered_img = to_image(img)
        path = '../paper_result/Figures/{}/'.format(model)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        for i in range(len(rendered_img)):
            imageio.imsave(os.path.join(path,'{}.png'.format(i)), rendered_img[i])

        m = load_mesh('../shapes/{}/mesh/mesh.obj'.format(model))
        m_ready = mesh.auto_normals(m)
        m_ready = mesh.compute_tangents(m_ready)

        handles = torch.load('../shapes/{}/{}/points.pt'.format(model, experiment), map_location='cuda:0')
        source_points2d = torch.stack(handles['handle'], dim=0) * (res // 512)
        target_points2d = torch.stack(handles['target'], dim=0) * (res // 512)

        uc3d = user_constraint_3d(m_ready, glctx, params_side, cfg)
        uc3d.reset(source_points2d, target_points2d)
        undeformed_img = render.render_mesh(glctx,
                                            m_ready.eval(params),
                                            params['mvp'],
                                            params['campos'],
                                            params['lightpos'],
                                            cfg['light_power'],
                                            res,
                                            spp=1,
                                            num_layers=cfg["layers"],
                                            msaa=False,
                                            background=bkg)

        undeformed_img = rgb_to_srgb(undeformed_img).permute(0, 3, 1, 2)[0]
        orig = undeformed_img.detach().clone().permute(1, 2, 0)
        orig = draw_points(orig, source_points2d)
        target_points_3d = uc3d.target_points_3d
        target_points_2d = project3to2(params, target_points_3d, False, cfg['train_res'], cfg['train_res'])
        orig = draw_points(orig, target_points_2d, [0, 0, 1])
        imageio.imsave(os.path.join(path, 'start.png'),
                       to_image(orig.permute(2, 0, 1)))

        path = '../paper_result/original_meshes/{}/'.format(model)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        obj.write_obj(path,m)




