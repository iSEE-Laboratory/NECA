import numpy as np
from lib.config import cfg
import cv2
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
import json
import torch

def farthest_point_sampling(points):
    """
    :param points: torch.Tensor, (N, D), D is the dimension of each point
    :return: (N, )
    """
    rest_indices = list(range(points.shape[0]))
    sampled_indices = [0]
    rest_indices.remove(sampled_indices[0])
    while len(rest_indices) > 0:
        rest_points = points[rest_indices]
        sampled_points = points[sampled_indices]
        dot_dist = torch.abs(torch.einsum('vi,mi->vm', rest_points, sampled_points))  # larger is closer
        neg_dot_dist = 1. - dot_dist  # smaller is closer
        min_dist = neg_dot_dist.min(1)[0]
        argmax_pos = min_dist.argmax().item()
        max_idx = rest_indices[argmax_pos]
        sampled_indices.append(max_idx)
        del rest_indices[argmax_pos]
    return sampled_indices
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def show_smpl(v, f):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


def load_cam(ann_file):
    if ann_file.endswith('.json'):
        annots = json.load(open(ann_file, 'r'))
        cams = annots['cams']['20190823']
    else:
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']

    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        K[i][:2] = K[i][:2] * cfg.image_ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT


def normalize(x):
    return x / np.linalg.norm(x)


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def gen_path(RT, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])
    # ipdb.set_trace()
    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, cfg.num_render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c


def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    return vertices, faces


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print("Saved:", filename)


def load_motion(path, separate_arms=True):
    motion_dict = dict(np.load(path))

    # The recurrent regressor is trained with 30fps sequences
    target_fps = 60
    drop_factor = int(motion_dict["mocap_framerate"] // target_fps)

    trans = motion_dict["trans"][::drop_factor]
    poses = motion_dict["poses"][::drop_factor, :72]
    shape = motion_dict["betas"][:10]

    # Separate arms
    if separate_arms:
        angle = 15
        left_arm = 17
        right_arm = 16

        poses = poses.reshape((-1, poses.shape[-1] // 3, 3))
        rot = R.from_euler('z', -angle, degrees=True)
        poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
        rot = R.from_euler('z', angle, degrees=True)
        poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

        poses = poses.reshape((poses.shape[0], -1))

    # Swap axes
    rotation = R.from_euler("zx", [-90, 270], degrees=True)
    root_rotation = R.from_rotvec(poses[:, :3])
    poses[:, :3] = (rotation * root_rotation).as_rotvec()
    trans = rotation.apply(trans)

    # Remove hand rotation
    poses[:, 66:] = 0

    # Center model in first frame
    trans = trans - trans[0]

    return {
        "pose": poses.astype(np.float32),
        "shape": shape.astype(np.float32),
        "translation": trans.astype(np.float32),
    }


def crop_mask_edge(mask):
    mask = mask.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    mask_erode = cv2.erode(mask.copy(), kernel)
    mask_dilate = cv2.dilate(mask.copy(), kernel)
    mask[(mask_dilate - mask_erode) == 1] = 100
    return mask


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents, return_joints=False):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    posed_joints = transforms[:, :3, 3].copy()

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    if return_joints:
        return transforms, posed_joints
    else:
        return transforms


def get_bounds(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= cfg.box_padding
    max_xyz += cfg.box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds
