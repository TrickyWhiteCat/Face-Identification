from mtcnn_model import P_Net
from train import train
from tensorflow import keras

def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #data path
    base_dir = r'C:/Users/data_/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with landmark
    model_path = r'C:/Users/data_/%s_model/PNet_landmark/PNet' % model_name
            
    prefix = model_path
    end_epoch = 30
    
    display = 100
    lr = 0.001
    train_PNet(base_dir, prefix, end_epoch, display, lr)
