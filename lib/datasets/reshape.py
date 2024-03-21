import numpy as np
import os
import cv2
from lib.config import cfg
import lib.utils.data_util as data_util
import lib.utils.cam_util as cam_util
import torch.utils.data as data
from lib.tools.smplmodel.body_model import SMPLlayer


class Dataset(data.Dataset):
    def __init__(self, data_root, human, split, begin_frame, end_frame, frame_interval, view):
        super(Dataset, self, ).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.frame_interval = frame_interval
        self.ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(self.ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        # current data info
        self.view = view
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][begin_frame:end_frame][::frame_interval]
        ]).ravel()

        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][begin_frame:end_frame][::frame_interval]
        ]).ravel()

        self.num_cams = len(view)
        all_params = np.load(os.path.join(os.getcwd(), 'data/params/{}.npy'.format(cfg.train_dataset.human[1:])),
                             allow_pickle=True).item()['pose']
        # train data info
        if self.split != 'train':
            self.cfg_train = cfg.train_dataset
            self.trained_params = all_params[
                                  self.cfg_train.begin_frame:self.cfg_train.end_frame][::self.cfg_train.frame_interval]
        else:
            # self.train_ims = self.ims
            self.trained_params = all_params[begin_frame:end_frame][::frame_interval]

        self.lbs_root = os.path.join(self.data_root, 'lbs')

        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)

        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)

        faces = np.load(os.path.join(self.lbs_root, 'faces.npy'))

        self.faces = faces.astype(np.long)
        self.big_A, self.big_joints, self.big_poses = self.load_bigpose()

        self.smpl = SMPLlayer('./data/smpl_model')

        params_path = os.path.join(self.data_root, 'new_params',
                                   '{}.npy'.format(1))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, 'new_params',
                                       '{:06d}.npy'.format(1))
        params = np.load(params_path, allow_pickle=True).item()
        self.shapes = params['shapes']
        tvertices, _, = self.smpl.forward(poses=self.big_poses[None], shapes=self.shapes)
        self.tvertices = tvertices[0].numpy()
        self.tbounds = data_util.get_bounds(self.tvertices)
        self.tnormal = data_util.compute_normal(self.tvertices, self.faces)
        vt, ft = self.load_texcoords(os.path.join(os.getcwd(), 'data/uvmapping.obj'))
        self.face_uv = vt[ft]
        self.ttb = self.get_tbn(self.faces, self.tvertices[self.faces], self.face_uv, self.tnormal)

        self.reshapes = cfg.reshape

        K, RT = data_util.load_cam(self.ann_file)
        self.K = K[0]
        self.RT = RT
        self.render_w2c = data_util.gen_path(self.RT)

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A, big_joints = data_util.get_rigid_transformation(
            big_poses, self.joints, self.parents, True)
        big_A = big_A.astype(np.float32)
        big_poses = big_poses.ravel().astype(np.float32)
        return big_A, big_joints.astype(np.float32), big_poses

    def load_texcoords(self, filename):
        vt, ft = [], []
        for content in open(filename):
            contents = content.strip().split(' ')

            if contents[0] == 'vt':
                vt.append([float(a) for a in contents[1:]])

            if contents[0] == 'f':
                ft.append([int(a.split('/')[1]) for a in contents[1:] if a])

        return np.array(vt, dtype='float64'), np.array(ft, dtype='int32') - 1

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

    def prepare_input(self, i, index, target_path=None):

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

        shapes[:, :2] += self.reshapes

        if target_path is not None:
            params_path = target_path
            params = np.load(params_path, allow_pickle=True).item()
            poses = params['poses'].astype(np.float32)

        alignment = None

        wxyz, A, canonical_joints = self.smpl.forward(poses=poses, shapes=shapes, Rh=Rh, Th=Th, return_joints=True,
                                                      alignment=alignment)

        shaped_txyz, shaped_bigA, shaped_big_joints = self.smpl.forward(poses=self.big_poses[None], shapes=shapes,
                                                                        return_joints=True, alignment=alignment)

        R_mat = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R_mat).astype(np.float32)
        poses = poses.ravel().astype(np.float32)
        return wxyz[0], pxyz[0], A[0].numpy(), Rh, R_mat, Th, poses, shapes, shaped_txyz[0].numpy(), shaped_bigA.numpy()

    def get_idx(self, imp):
        img_path = os.path.join(self.data_root, imp)
        if self.human in ['C313', 'C315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        return i, frame_index

    def __getitem__(self, index):
        i, frame_index = self.get_idx(self.ims[0])
        wvertices, pvertices, A, Rh, R_mat, Th, poses, shapes, shaped_tvertices, shaped_big_A = self.prepare_input(i,
                                                                                                                   index)

        pnormal = data_util.compute_normal(pvertices, self.faces)
        ptb = self.get_tbn(self.faces, pvertices[self.faces], self.face_uv, pnormal)
        shaped_tnormal = data_util.compute_normal(shaped_tvertices, self.faces)
        shaped_ttb = self.get_tbn(self.faces, shaped_tvertices[self.faces], self.face_uv, shaped_tnormal)
        pbounds = data_util.get_bounds(pvertices)
        shaped_tbounds = data_util.get_bounds(shaped_tvertices)

        wbounds = data_util.get_bounds(wvertices.numpy())

        RT = self.render_w2c[index]
        K = self.K
        H = int(cfg.H * cfg.image_ratio)
        W = int(cfg.W * cfg.image_ratio)

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
            'big_A': shaped_big_A,
            'poses': poses,
            'shapes': shapes,
            'weights': self.weights,
            'faces': self.faces,

            'ttb': self.ttb,
            'shaped_ttb': shaped_ttb,
            'ptb': ptb,

            'tvertices': self.tvertices,
            'shaped_tvertices': shaped_tvertices,
            'pvertices': pvertices,

            'pbounds': pbounds,
            'tbounds': self.tbounds,
            'shaped_tbounds': shaped_tbounds,
        }
        ret.update(meta)

        # transformation
        meta = {'K': K, 'E': RT, 'R': R_mat, 'Rh': Rh, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        meta = {
            'frame_index': index,
            'latent_index': frame_index - self.begin_frame
        }

        ret.update(meta)

        ret.update({'all_poses': self.trained_params})
        return ret

    def __len__(self):
        return len(self.render_w2c)
