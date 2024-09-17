import pickle
from PIL import Image
pk_path = "E:/Dataset/sunrgbd_train_test_data/1.pkl"
with open(pk_path, 'rb') as f:
    image_dict = pickle.load(f)
    image_array = image_dict['rgb_img']
    image = Image.fromarray(image_array)
    image.show()


