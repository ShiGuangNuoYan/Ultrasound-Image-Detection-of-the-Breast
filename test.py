import pickle
pk_path = "E:/Dataset/sunrgbd_train_test_data/1.pkl"
with open(pk_path, 'rb') as f:
    image_array = pickle.load(f)
    print(image_array)
