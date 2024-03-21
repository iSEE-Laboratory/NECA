import numpy as np
import os
import cv2
from lib.config import cfg
import lib.utils.data_util as data_util
import lib.utils.cam_util as cam_util
from lib.datasets.base_dataset import BaseDataset
from lib.tools.smplmodel.body_model import SMPLlayer


class Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super(Dataset, self, ).__init__(**kwargs)
        self.tnormal = data_util.compute_normal(self.tvertices, self.faces)
        vt, ft = self.load_texcoords(os.path.join(os.getcwd(), 'data/uvmapping.obj'))
        self.face_uv = vt[ft]
        self.ttb = self.get_tbn(self.faces, self.tvertices[self.faces], self.face_uv, self.tnormal)

        K, RT = data_util.load_cam(self.ann_file)

        self.K = K[0]
        self.RT = RT
        self.render_w2c = data_util.gen_path(self.RT)

        self.smpl = SMPLlayer('./data/smpl_model')

    def load_texcoords(self, filename):
        vt, ft = [], []
        for content in open(filename):
            contents = content.strip().split(' ')

            if contents[0] == 'vt':
                vt.append([float(a) for a in contents[1:]])

            if contents[0] == 'f':
                ft.append([int(a.split('/')[1]) for a in contents[1:] if a])

        return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1

    def prepare_input(self, i, target_path=None):

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, 'new_params',
                                   '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, 'new_params',
                                       '{:06d}.npy'.format(i))

        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        # calculate the skeleton transformation
        poses = params['poses'].astype(np.float32)
        shapes = params['shapes'].astype(np.float32)

        wxyz, A, canonical_joints = self.smpl.forward(poses=poses, shapes=shapes, Rh=Rh, Th=Th, return_joints=True,
                                                      alignment=None)

        R_mat = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R_mat).astype(np.float32)
        poses = poses.ravel().astype(np.float32)
        return wxyz[0].numpy(), pxyz[0], A[0].numpy(), Rh, R_mat, Th, poses, shapes

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

        # b = mat_tb[:, 1, :]
        # b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        # bitan = np.zeros((6890, 3))
        # bitan[f[:, 0]] += b
        # bitan[f[:, 1]] += b
        # bitan[f[:, 2]] += b
        # bitan = bitan / (np.linalg.norm(bitan, axis=-1, keepdims=True) + 1e-9)
        tan = (tan - np.sum(tan * normal, axis=-1, keepdims=True) * normal)
        tan = tan / (np.linalg.norm(tan, axis=-1, keepdims=True) + 1e-9)
        bitan = np.cross(normal, tan)
        return np.stack([tan, bitan, normal], axis=-2)

    def __len__(self):
        return len(self.render_w2c)

    def __getitem__(self, index):

        i, frame_index = self.get_idx(self.ims[index // len(self.render_w2c)])

        K = self.K

        RT = self.render_w2c[index]

        wvertices, pvertices, A, Rh, R_mat, Th, poses, shapes = self.prepare_input(i)

        pnormal = data_util.compute_normal(pvertices, self.faces)
        ptb = self.get_tbn(self.faces, pvertices[self.faces], self.face_uv, pnormal)

        pbounds = data_util.get_bounds(pvertices)
        wbounds = data_util.get_bounds(wvertices)

        ray_o, ray_d, near, far, center, scale, mask_at_box = cam_util.image_rays(
            RT, K, wbounds)

        ret = {
            'ray_o': ray_o,  # (N,3)
            'ray_d': ray_d,  # (N,3)
            'near': near,  # (N,)
            'far': far,  # (N,)
        }

        ret.update({"mask_at_box": mask_at_box})  # (N,)

        # blend skinning
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'shapes': shapes,

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
        H = int(cfg.H * cfg.image_ratio)
        W = int(cfg.W * cfg.image_ratio)
        # transformation

        meta = {'R': R_mat, 'Rh': Rh, 'Th': Th, 'H': H, 'W': W, 'E': self.RT[0][:3, :].astype(np.float32)}
        ret.update(meta)

        meta = {
            'frame_index': index,
            'cam_index': index,
            'latent_index': frame_index
        }

        ret.update(meta)

        ret.update({'all_poses': self.trained_params})

        return ret
