import os
import numpy as np
import tqdm


def get_params(params_path):
    params = np.load(os.path.join(data_root, 'new_params', params_path), allow_pickle=True).item()

    Rh = params['Rh'].astype(np.float32)
    Th = params['Th'].astype(np.float32)
    poses = params['poses'].astype(np.float32)
    shapes = params['shapes'].astype(np.float32)

    return Rh, Th, poses, shapes


def main(subject_id):
    Rhs = []
    Ths = []
    poses = []
    shapes = []
    file_lst = os.listdir(os.path.join(data_root, 'new_params'))
    file_lst.sort(key=lambda x: int(x.split('.')[0]))
    for f in tqdm.tqdm(file_lst):
        Rh, Th, pose, shape = get_params(f)
        Rhs.append(Rh)
        Ths.append(Th)
        poses.append(pose)
        shapes.append(shape)

    Rh = np.concatenate(Rhs, axis=0)
    Th = np.concatenate(Ths, axis=0)
    pose = np.concatenate(poses, axis=0)
    shape = np.concatenate(shapes, axis=0)

    params = {
        'Rh': Rh,
        'Th': Th,
        'pose': pose,
        'shape': shape
    }

    os.makedirs(os.path.join(os.getcwd(), 'data/params'), exist_ok=True)
    np.save(os.path.join(os.getcwd(), 'data/params/{}.npy'.format(subject_id)), params)


if __name__ == "__main__":

    subject_ids = ['313', '315', '377', '386', '387', '390', '392', '393', '394']

    for subject_id in subject_ids:
        data_root = './data/zju_mocap/CoreView_{}/'.format(subject_id)
        main(subject_id)
