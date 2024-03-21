import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
import trimesh
from lib.config import cfg
import lib.utils.data_util as data_util

class BaseDataset(data.Dataset):

    def __init__(self, data_root, human, split, begin_frame, end_frame, frame_interval, view):
        super(BaseDataset, self).__init__()

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
        self.cfg_train = cfg.train_dataset
        self.trained_params = all_params[
                              self.cfg_train.begin_frame:self.cfg_train.end_frame][::self.cfg_train.frame_interval]

        self.lbs_root = os.path.join(self.data_root, 'lbs')

        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)

        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)

        faces = np.load(os.path.join(self.lbs_root, 'faces.npy'))

        self.faces = faces.astype(np.int64)
        self.big_A, self.big_joints, self.big_poses = self.load_bigpose()

        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        self.tvertices = np.load(vertices_path).astype(np.float32)

        self.tpose_mesh = self.tvertices[self.faces]
        self.smpl_mesh = trimesh.Trimesh(self.tvertices, self.faces)
        self.tbounds = data_util.get_bounds(self.tvertices)

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

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, 'new_vertices',
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, 'new_vertices',
                                         '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, 'new_params',
                                   '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, 'new_params',
                                       '{:06d}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()

        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R_mat = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R_mat).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)

        shapes = params['shapes'][0].astype(np.float32)
        joints = self.joints
        parents = self.parents
        A, canonical_joints = data_util.get_rigid_transformation(
            poses, joints, parents, return_joints=True)

        posed_joints = np.dot(canonical_joints, R_mat.T) + Th

        poses = poses.ravel().astype(np.float32)

        return wxyz, pxyz, A, Rh, R_mat, Th, poses, shapes, canonical_joints.astype(np.float32)

    def get_mask(self, ims, index):

        mask_cihp_path = os.path.join(self.data_root, 'mask_cihp',
                                      ims[index])[:-4] + '.png'

        mask_path = os.path.join(self.data_root, 'mask',
                                 ims[index])[:-4] + '.png'

        mask = None

        if os.path.exists(mask_path):
            mask = imageio.imread(mask_path)
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            mask = (mask != 0).astype(np.uint8)

        if os.path.exists(mask_cihp_path):
            mask_cihp = imageio.imread(mask_cihp_path)
            if len(mask_cihp.shape) == 3:
                mask_cihp = mask_cihp[..., 0]
            mask_cihp = (mask_cihp != 0).astype(np.uint8)
            if mask is not None:
                mask = (mask | mask_cihp).astype(np.uint8)
            else:
                mask=mask_cihp.astype(np.uint8)

        orig_mask = mask.copy()

        if cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            mask_erode = cv2.erode(mask.copy(), kernel)
            mask_dilate = cv2.dilate(mask.copy(), kernel)
            mask[(mask_dilate - mask_erode) == 1] = 100

        return mask, orig_mask

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
        pass

    def __len__(self):
        return len(self.ims)
