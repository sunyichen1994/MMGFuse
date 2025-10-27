class args():

    # training args
    g1 = 0.5
    epochs = 50  # number of training epochs, default is 2
    batch_size = 16  # batch size for training, default is 4
    dataset2 = r"/media/sunyichen/HIKSEMI/MSRS-main/train/ir"
    train_num = 40000

    HEIGHT = 256
    WIDTH = 256
    save_model_dir = "pretrained"  # path to folder where trained model will be saved.
    save_loss_dir = "pretrained/loss"  # path to folder where trained model will be saved.

    image_size = 256  # size of training images, default is 256 X 256
    cuda = 1  # set it to 1 for running on GPU, 0 for CPU
    seed = 42  # random seed for training
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
    alpha = 0.5
    beta = 0.5
    gama = 1
    yita = 1
    deta = 1

    lr = 1e-4  # learning rate, default is 0.001
    lr_d = 1e-4
    log_interval = 10  # number of images after which the training loss is logged, default is 500
    resume = None
    trans_model_path = None
    is_para = False