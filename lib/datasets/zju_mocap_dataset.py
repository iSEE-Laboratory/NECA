import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
import lib.utils.data_util as data_util
import lib.utils.cam_util as cam_util
from lib.datasets.base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super(Dataset, self, ).__init__(**kwargs)

        self.tnormal = data_util.compute_normal(self.tvertices, self.faces)
        vt, ft = self.load_texcoords(os.path.join(os.getcwd(), 'data/uvmapping.obj'))
        self.face_uv = vt[ft]
        self.ttb = self.get_tbn(self.faces, self.tvertices[self.faces], self.face_uv, self.tnormal)
        self.ray_mode = cfg.ray_mode

    def update_ray_mode(self, mode):
        self.ray_mode = mode

    def load_texcoords(self, filename):
        vt, ft = [], []
        for content in open(filename):
            contents = content.strip().split(' ')

            if contents[0] == 'vt':
                vt.append([float(a) for a in contents[1:]])

            if contents[0] == 'f':
                ft.append([int(a.split('/')[1]) for a in contents[1:] if a])

        return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1

    def process_img(self, ims, index, cam_index):

        img_path = os.path.join(self.data_root, ims[index])

        img = imageio.imread(img_path).astype(np.float32) / 255.

        mask, orig_mask = self.get_mask(ims, index)

        H, W = img.shape[:2]
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_mask = cv2.resize(orig_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        K = np.array(self.cams['K'][cam_index])
        D = np.array(self.cams['D'][cam_index])
        img = cv2.undistort(img, K, D)
        mask = cv2.undistort(mask, K, D)
        orig_mask = cv2.undistort(orig_mask, K, D)

        R = np.array(self.cams['R'][cam_index])
        T = np.array(self.cams['T'][cam_index]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.image_ratio), int(img.shape[1] * cfg.image_ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_mask = cv2.resize(orig_mask, (W, H),
                               interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            img[mask == 0] = cfg.bg_color[0]
        K[:2] = K[:2] * cfg.image_ratio
        return H, W, img, mask, orig_mask, K, R, T

    def get_tbn(self, f, faces, face_uv, normal):
        v0, v1, v2 = faces[..., 0], faces[..., 1], faces[..., 2]
        e1 = v1 - v0
        e2 = v2 - v0
        uv0, uv1, uv2 = face_uv[:, 0, :], face_uv[:, 1, :], face_uv[:, 2, :]
        delta_uv1 = uv1 - uv0
        delta_uv2 = uv2 - uv0
        mat_e = np.stack([e1, e2], axis=-2)
        mat_deltauv = np.stack([delta_uv1, delta_uv2], axis=-2)
        mat_tb = np.matmul(np.linalg.inv(mat_deltauv), mat_e)
        t = mat_tb[:, 0, :]
        t = t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-9)
        tan = np.zeros((6890, 3))
        tan[f[:, 0]] += t
        tan[f[:, 1]] += t
        tan[f[:, 2]] += t
        tan = tan / (np.linalg.norm(tan, axis=-1, keepdims=True) + 1e-9)

        tan = (tan - np.sum(tan * normal, axis=-1, keepdims=True) * normal)
        tan = tan / (np.linalg.norm(tan, axis=-1, keepdims=True) + 1e-9)
        bitan = np.cross(normal, tan)
        return np.stack([tan, bitan, normal], axis=-2)

    def __getitem__(self, index):

        i, frame_index = self.get_idx(self.ims[index])

        cam_index = self.cam_inds[index]
        H, W, img, mask, orig_mask, K, R, T = self.process_img(self.ims, index, cam_index)

        wvertices, pvertices, A, Rh, R_mat, Th, poses, shapes, posed_joints = self.prepare_input(i)

        pnormal = data_util.compute_normal(pvertices, self.faces)
        ptb = self.get_tbn(self.faces, pvertices[self.faces], self.face_uv, pnormal)

        pbounds = data_util.get_bounds(pvertices)
        wbounds = data_util.get_bounds(wvertices)

        if cfg.erode_edge:
            orig_mask = data_util.crop_mask_edge(orig_mask)

        if self.ray_mode == 'image' or self.split == 'test':
            split = self.split

            rgb, ray_o, ray_d, near, far, coord, mask_at_box = \
                cam_util.sample_ray(img, mask, K, R, T, wbounds, cfg.num_rays, split)
            mask = mask[coord[:, 0], coord[:, 1]]

            occupancy = orig_mask[coord[:, 0], coord[:, 1]]

        elif self.ray_mode == 'patch':

            ray_o, ray_d, rgb, mask, occupancy, near, far, \
                target_patches, patch_masks, patch_div_indices = cam_util.sample_patch(img, mask, orig_mask, wbounds, K,
                                                                                       R,
                                                                                       T)

            near = near[:, 0]
            far = far[:, 0]
            ray_o = ray_o.astype(np.float32)
            ray_d = ray_d.astype(np.float32)
        else:
            raise Exception('')
        ret = {
            'img': img,
            'rgb': rgb,  # (N,3)
            'occ': occupancy,  # (N,)
            'ray_o': ray_o,  # (N,3)
            'ray_d': ray_d,  # (N,3)
            'near': near,  # (N,)
            'far': far,  # (N,)
        }
        if self.ray_mode == 'image' or self.split == 'test':
            ret.update({"mask_at_box": mask_at_box})  # (N,)
        elif self.ray_mode == 'patch':
            ret.update({
                "target_patches": target_patches,
                "patch_masks": patch_masks,
                "patch_div_indices": patch_div_indices
            })

        # blend skinning
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'shapes': shapes,
            'posed_joints': posed_joints,
            'big_joints': self.big_joints,
            'weights': self.weights,

            'faces': self.faces,
            'pnormal': pnormal,

            'ttb': self.ttb,
            'ptb': ptb,
            'tpose_mesh': self.tpose_mesh,
            'tvertices': self.tvertices,
            'pvertices': pvertices,
            'wvertices': wvertices,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': self.tbounds,
        }
        ret.update(meta)

        E = np.concatenate([R, T], axis=-1)
        # transformation
        meta = {'K': K, 'E': E.astype(np.float32), 'R': R_mat, 'Rh': Rh, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        meta = {
            'frame_index': frame_index,
            'cam_index': cam_index,
            'latent_index': frame_index - self.begin_frame
        }

        ret.update(meta)

        ret.update({'all_poses': self.trained_params})

        return ret

