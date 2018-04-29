from skimage.feature import *
from skimage import color, filters
from skimage.transform import resize
import numpy as np
from PIL import Image
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input


from fonduer import SnorkelSession

session = SnorkelSession()

from fonduer import DetailedImage, ImageFeatures, Context

def gen_image_features(docs_path=None, clear=True):
    if clear:
        session.query(Context).filter(Context.stable_id.like('%feature%')).delete(synchronize_session=False)
    if docs_path == None:
        print("Specify html document path")
        return
    print("Start generating image features!")
    model = ResNet50(weights='imagenet', include_top=False)
    IMG_SZ=224
    figures = session.query(DetailedImage).all()
    print("Number of figures:", len(figures))
    for figure in figures:
        img_path = docs_path + figure.url
        img = Image.open(img_path)
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
        feature_idx = 0
        # Add HOG features
        res = hog(img, block_norm='L2-Hys', pixels_per_cell=(10, 10), cells_per_block=(2,2))
        feature_idx = add_feats(res, "HOG", figure, feature_idx)
        # Add Local Binary Pattern features
        res = local_binary_pattern(img, P=9, R=5)
        lbp_hist = np.histogram(res, bins=250)[0]
        feature_idx = add_feats(lbp_hist, "LBP", figure, feature_idx)
        # Add binary image
        thres = filters.threshold_otsu(img)
        bw_img = np.array(img<thres, dtype=np.float32)
        bw_small = resize(bw_img, (50,50), mode='reflect').flatten()
        feature_idx = add_feats(bw_small, "Binary", figure, feature_idx)
        # Add ResNet feature
        img_data = np.array((img, img, img), dtype=np.float32).transpose((1, 2, 0))
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        resnet_feature = model.predict(img_data)
        res = resnet_feature.flatten()
        feature_idx = add_feats(res,"CNN", figure, feature_idx)
    session.commit()
    print("Total {} type of image features added.".format(feature_idx))

def add_feats(feature, feature_type, figure, feature_idx):
    test_feature = pickle.dumps(feature)
    stable_id = "%s::%s:%s:%s" % \
                (figure.document.name, figure.name, "feature", feature_type)
    figure_feature_one = ImageFeatures(
        image=figure,
        stable_id=stable_id,
        position=feature_idx,
        description=feature_type,
        features=test_feature
    )
    session.add(figure_feature_one)
    return feature_idx+1
