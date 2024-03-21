import numpy as np
import torch
import torch.nn as nn
from lib.config import cfg
import os.path as osp


def read_hdr(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    import cv2

    if cfg.tonemapping:
        img = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH)
        tonemapDurand = cv2.createTonemapReinhard(2.2, 0, 0, 0)
        ldrDurand = tonemapDurand.process(img)
        im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
        rgb = cv2.cvtColor(im2_8bit, cv2.COLOR_BGR2RGB) / 255.
        rgb = cv2.resize(rgb, (32, 16))
    else:
        with open(path, 'rb') as h:
            buffer_ = np.fromstring(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (32, 16))

    return rgb


def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print(
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians")


def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    # pts_cart = np.array(pts_cart)

    # Validate inputs
    # is_one_point = False
    # if pts_cart.shape == (3,):
    #     is_one_point = True
    #     pts_cart = pts_cart.reshape(1, 3)
    # elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
    #     raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r

    r = torch.sqrt(torch.sum(torch.square(pts_cart), dim=-1))

    # Compute latitude
    z = pts_cart[..., 2]
    lat = torch.arcsin(z / r)

    # Compute longitude
    x = pts_cart[..., 0]
    y = pts_cart[..., 1]
    lng = torch.arctan2(y, x)  # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = torch.stack((r, lat, lng), dim=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    # if is_one_point:
    #     pts_sph = pts_sph.reshape(3)

    return pts_sph


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h)
    lng_step_size = 2 * np.pi / (envmap_w)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas


class EnvMap(nn.Module):
    def __init__(self, light_h=16, hdr_path=''):
        super(EnvMap, self).__init__()

        self.light_h = light_h
        lxyz, lareas = self.gen_lights()
        self.register_buffer('light_xyz', lxyz)
        self.register_buffer('lareas', lareas)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.light_res = (light_h, 2 * light_h)
        light = torch.randn(self.light_res + (1,))
        self.light = nn.Parameter(light, requires_grad=True)
        self.hdr_path = hdr_path

        if hdr_path != '' and hdr_path != 'self':
            arr = read_hdr(osp.join('light-probes/', hdr_path))
            novel_probes = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            self.novel_probes = novel_probes

    def get_lxyz(self):
        return self.light_xyz

    def gen_lights(self):
        light_h = int(self.light_h)
        light_w = int(light_h * 2)
        lxyz, lareas = gen_light_xyz(light_h, light_w)
        if 'zju' in cfg.train_dataset.data_root:
            print("zju-mocap, using different lighting......")
            lxyz[..., 1] = -lxyz[..., 1]
            lxyz[..., 0] = -lxyz[..., 0]
        else:
            lxyz = lxyz[..., [0, 2, 1]]
            lxyz[..., 1] = -lxyz[..., 1]
        # lxyz = lxyz[..., [0, 2, 1]]
        # lxyz[..., 1] = -lxyz[..., 1]
        lxyz = torch.from_numpy(lxyz).float()
        lareas = torch.from_numpy(lareas).float()
        return lxyz, lareas

    def get_light_energy(self):
        return torch.sigmoid(self.light)

    def forward(self, visibility, surf, normal, front_light=None, lrot=None):

        light = self.get_light_energy()

        if self.hdr_path != '' and self.hdr_path != 'self':
            probe = self.novel_probes
            if cfg.light_scale:
                if 'zju' in cfg.train_dataset.data_root:
                    light = 0.31 * light + probe
                else:
                    light = probe

        light = light.reshape(1, -1)

        lxyz = self.get_lxyz().reshape(1, -1, 3)
        if lrot is not None:
            lxyz = torch.matmul(lxyz, lrot)

        surf2light = lxyz[:, None] - surf[:, :, None]
        surf2light = surf2light / (torch.norm(surf2light, dim=-1, keepdim=True) + 1e-9)
        areas = self.lareas.reshape(1, -1, 1)

        if front_light is not None:
            light = light[front_light][None]
            areas = areas[front_light][None]

        lcos = torch.einsum('bijk,bik->bij', surf2light, normal)

        front_lit = lcos > 0
        if visibility is not None:
            front_lit = visibility[None, :, :, None] * front_lit[..., None]
        else:
            front_lit = front_lit[..., None]

        def integrate(light):
            if front_light is not None:
                light_flat = light.reshape(front_light.sum(), -1)
            else:
                light_flat = light.reshape(512, -1)

            light = front_lit * light_flat[None, None, :]  # NxLx3

            light_pix_contrib = light * lcos[..., None] * areas[None]  # NxLx3
            rgb = torch.sum(light_pix_contrib, dim=-2)  # Nx3

            return rgb

        rgb = integrate(light)
        return rgb
