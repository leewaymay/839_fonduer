from skimage.feature import *
from skimage import color
import numpy as np
from PIL import Image
import pickle

from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import DetailedImage, ImageFeatures

def gen_image_features(docs_path=None):
    if docs_path == None:
        print("Specify html document path")
        return
    print("Start generating image features!")
    IMG_SZ=200
    figures = session.query(DetailedImage).all()
    print("Number of figures:", len(figures))
    for idx in range(len(figures)):
        img = Image.open(docs_path + figures[idx].url)
        if img.size[0] < img.size[1]:
            scale = IMG_SZ/img.size[1]
            s_sz = int(img.size[0]*scale)
            before = (IMG_SZ-s_sz)//2
            sz = (s_sz, IMG_SZ)
            img = np.asarray(img.resize(sz, Image.ANTIALIAS))
            img = np.pad(img, ((0,0), (before, IMG_SZ-s_sz-before)), 'maximum')
        else:
            scale = IMG_SZ / img.size[0]
            s_sz = int(img.size[1] * scale)
            sz = (IMG_SZ, s_sz)
            before = (IMG_SZ - s_sz) // 2
            img = np.asarray(img.resize(sz, Image.ANTIALIAS))
            img = np.pad(img, ((before, IMG_SZ - s_sz - before), (0, 0)), 'maximum')

        img = color.rgb2gray(img)
        res = hog(img, block_norm='L2-Hys', pixels_per_cell=(12, 12), cells_per_block=(2,2))
        test_feature = pickle.dumps(res)
        feature_type = "HOG"
        stable_id = "%s::%s:%s:%s" % \
                        (figures[idx].document.name, figures[idx].name, "feature", feature_type)
        feature_idx = 0
        figure_feature_one = ImageFeatures(
                        image=figures[idx],
                        stable_id=stable_id,
                        position=feature_idx,
                        description = feature_type,
                        features=test_feature
                    )
        session.add(figure_feature_one)
    session.commit()

    print("Finished!")



