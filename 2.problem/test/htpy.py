import h5py
import numpy as np

hf = h5py.File(f'../datab/file.hdf5','r')
sal_maps = np.array(hf['saliencys'])
print(sal_maps.shape)

# sal_maps =np.array([], dtype=np.float32).reshape((0,) + img_size)
# probs = np.array([], dtype=np.float32)
# preds = np.array([], dtype=np.uint8)

# sal_maps = np.vstack([sal_maps,sal_maps_b])
# probs = np.append(probs, probs_b)
# preds = np.append(preds,preds_b)

# sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
# probs = np.array([], dtype=np.float32)
# preds = np.array([], dtype=np.uint8)
# gc.collect()

# del sal_maps, probs, preds

sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
probs = np.array([], dtype=np.float32)
preds = np.array([], dtype=np.uint8)
gc.collect()
# sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
# probs = np.array([], dtype=np.float32)
# preds = np.array([], dtype=np.uint8)f
# print('Save_saliency_maps::',idx)
# if idx >= 0 and idx <3000:continue
# else: break