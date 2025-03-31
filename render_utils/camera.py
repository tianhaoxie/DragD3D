#from CLIP-Mesh: https://github.com/NasirKhalid24/CLIP-Mesh.git

import glm
import torch
import random
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

from .resize_right import resize

blurs = [
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(2, 2))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(2, 2))
    ]),
]


def get_random_bg(h, w):
    p = torch.rand(1)

    if p > 0.66666:
        background = blurs[random.randint(0, 3)](torch.rand((1, 3, h, w))).permute(0, 2, 3, 1)
    elif p > 0.333333:
        size = random.randint(5, 10)
        background = torch.vstack([
            torch.full((1, size, size), torch.rand(1).item() / 2),
            torch.full((1, size, size), torch.rand(1).item() / 2),
            torch.full((1, size, size), torch.rand(1).item() / 2),
        ]).unsqueeze(0)

        second = torch.rand(3)

        background[:, 0, ::2, ::2] = second[0]
        background[:, 1, ::2, ::2] = second[1]
        background[:, 2, ::2, ::2] = second[2]

        background[:, 0, 1::2, 1::2] = second[0]
        background[:, 1, 1::2, 1::2] = second[1]
        background[:, 2, 1::2, 1::2] = second[2]

        background = blurs[random.randint(0, 3)](resize(background, out_shape=(h, w)))

        background = background.permute(0, 2, 3, 1)

    else:
        background = torch.vstack([
            torch.full((1, h, w), torch.rand(1).item()),
            torch.full((1, h, w), torch.rand(1).item()),
            torch.full((1, h, w), torch.rand(1).item()),
        ]).unsqueeze(0).permute(0, 2, 3, 1)

    return background


def cosine_sample(N: np.ndarray) -> np.ndarray:
    """
    #----------------------------------------------------------------------------
    # Cosine sample around a vector N
    #----------------------------------------------------------------------------

    Copied from nvdiffmodelling

    """
    # construct local frame
    N = N / np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0, dx0) > np.dot(dx1, dx1) else dx1
    dx = dx / np.linalg.norm(dx)
    dy = np.cross(N, dx)
    dy = dy / np.linalg.norm(dy)

    # cosine sampling in local frame
    phi = 2.0 * np.pi * np.random.uniform()
    s = np.random.uniform()
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi) * sintheta
    y = np.sin(phi) * sintheta
    z = costheta

    # local to world
    return dx * x + dy * y + N * z


def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]],dtype=np.float32)

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def get_camera_params(elev_angle, azim_angle, distance, resolution, bkgs=None, device='cuda:0', fov=60, look_at=[0, 0, 0], up=[0, -1, 0]):
    elev = np.radians(elev_angle)
    azim = np.radians(azim_angle)

    cam_z = distance * np.cos(elev) * np.cos(azim)
    cam_y = distance * np.sin(elev)
    cam_x = distance * np.cos(elev) * np.sin(azim)

    modl = glm.mat4()
    view = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )

    a_mv = view * modl
    a_mv = np.array(a_mv.to_list()).T
    proj_mtx = persp_proj(fov)

    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]

    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = a_lightpos
    if bkgs is None:
        bkgs = torch.ones(resolution, resolution, 3).unsqueeze(0)

    return {
        'mvp': torch.from_numpy(a_mvp).float().to(device),
        'lightpos': torch.from_numpy(a_lightpos).float().to(device),
        'campos': torch.from_numpy(a_campos).float().to(device),
        'bkgs': bkgs.to(device)
    }



# Returns a batch of camera parameters
class CameraBatch(torch.utils.data.Dataset):
    def __init__(
            self,
            image_resolution,
            distances,
            azimuths,
            elevation_params,
            fovs,
            aug_loc,
            aug_light,
            aug_bkg,
            bs,
            look_at=[0, 0, 0], up=[0, -1, 0]
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]

        self.elev_alpha = elevation_params[0]
        self.elev_beta = elevation_params[1]
        self.elev_max = elevation_params[2]

        self.aug_loc = aug_loc
        self.aug_light = aug_light
        self.aug_bkg = aug_bkg

        self.look_at = look_at
        self.up = up

        self.batch_size = bs

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):

        elev = np.radians(np.random.beta(self.elev_alpha, self.elev_beta) * self.elev_max)
        #elev = np.radians(np.random.uniform(0,self.elev_max))
        #azim = np.radians(np.random.uniform(self.azim_min, self.azim_max + 1.0))
        azim_d = np.random.uniform(self.azim_min, self.azim_max)
        azim = np.radians(azim_d)
        dist = np.random.uniform(self.dist_min, self.dist_max)
        fov = np.random.uniform(self.fov_min, self.fov_max)

        proj_mtx = persp_proj(fov)

        # Generate random view
        cam_z = dist * np.cos(elev) * np.cos(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.sin(azim)

        if self.aug_loc:

            # Random offset
            limit = self.dist_min // 2
            rand_x = np.random.uniform(-limit, limit)
            rand_y = np.random.uniform(-limit, limit)

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:

            modl = glm.mat4()

        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos = np.linalg.inv(r_mv)[:3, 3]

        if self.aug_light:
            lightpos = cosine_sample(campos) * dist
        else:
            lightpos = campos * dist

        if self.aug_bkg:
            bkgs = get_random_bg(self.res, self.res).squeeze(0)
        else:
            bkgs = torch.zeros(self.res, self.res, 3)

        return {
            'azim': torch.tensor([azim_d]).float(),
            'mvp': torch.from_numpy(mvp).float(),
            'lightpos': torch.from_numpy(lightpos).float(),
            'campos': torch.from_numpy(campos).float(),
            'bkgs': bkgs
        }

class OrbitCamera:
    def __init__(self, W, H, r=2, fov = 60, near=0.1, far=50 ):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fov = fov  # in degree
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def pose(self):

        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        #res[1,1] = -1
        return res

    @property
    def view(self):

        return np.linalg.inv(self.pose)

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fov) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def perspective(self):
        return persp_proj(self.fov,self.W/self.H,self.near,self.far)

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(0.05 * dx)
        rotvec_y = side * np.radians(0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.00004 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

    def get_params(self):

        mv = self.view
        proj = self.perspective
        mvp = np.matmul(proj, mv).astype(np.float32)[None, ...]
        campos = self.pose[None,:3,3]
        lightpos = campos
        return {
            'mvp': mvp,
            'lightpos': lightpos,
            'campos': campos,
        }


class EasyCamera():
    def __init__(self, res):
        self.azim = 0
        self.elev = 0
        self.cam_dis = 2
        self.res = res

    def get_params(self):
        params = get_camera_params(self.elev,self.azim,self.cam_dis,self.res)
        for key in params:
            params[key] = params[key].detach().cpu().numpy()
        return params

    def change_view(self,azim,elev,cam_dis):
        self.azim = azim
        self.elev = elev
        self.cam_dis = cam_dis
