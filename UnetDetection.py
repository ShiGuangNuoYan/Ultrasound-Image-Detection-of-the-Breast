# 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from model.Unet import Fcrn_encode,Fcrn_decode
from model.GAN import Generator,Discriminator
from Dataset.datasets import GAN_Dataset,test_Dataset
import cv2
import argparse   # argparse库: 解析命令行参数
from tqdm import tqdm   # 进度条


# 创建一个解析对象
parser = argparse.ArgumentParser(description="Choose mode")
# 输入命令行和参数
parser.add_argument('-mode', required=True, choices=['train', 'test'], default='train')
parser.add_argument('-dim', type=int, default=16)
parser.add_argument('-num_epochs', type=int, default=3)
parser.add_argument('-image_scale_h', type=int, default=256)
parser.add_argument('-image_scale_w', type=int, default=256)
parser.add_argument('-batch', type=int, default=4)
parser.add_argument('-img_cut', type=int, default=4)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-lr_1', type=float, default=5e-5)
parser.add_argument('-alpha', type=float, default=0.05)
parser.add_argument('-sa_scale', type=float, default=8)
parser.add_argument('-latent_size', type=int, default=100)
parser.add_argument('-data_path', type=str, default='./munich/train/img')
parser.add_argument('-label_path', type=str, default='./munich/train/lab')
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-load_model', required=True, choices=['True', 'False'], help='choose True or False', default='False')

# parse_args()方法进行解析
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
use_cuda = torch.cuda.is_available()
print("use_cuda:", use_cuda)

# 指定计算机的第一个设备是GPU
device = torch.device("cuda" if use_cuda else "cpu")
IMG_CUT = opt.img_cut
LATENT_SIZE = opt.latent_size
writer = SummaryWriter('./runs2/gx0102')

# 创建文件路径
def auto_create_path(FilePath):
    if os.path.exists(FilePath):
            print(FilePath + ' dir exists')
    else:
            print(FilePath + ' dir not exists')
            os.makedirs(FilePath)

# 创建文件存放训练的结果
auto_create_path('./test/lab_dete_AVD')
auto_create_path('./model')
auto_create_path('./results')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img_road = GAN_Dataset(opt,transform)
train_dataloader = DataLoader(img_road, batch_size=opt.batch, shuffle=True)
print(len(train_dataloader.dataset), train_dataloader.dataset[7][1].shape)

img_road_test = test_Dataset(opt,transform)

test_dataloader = DataLoader(img_road_test, batch_size=1, shuffle=False)

print(len(test_dataloader.dataset), test_dataloader.dataset[7][1].shape)

loss = nn.BCELoss()

fcrn_encode = Fcrn_encode(opt)
fcrn_encode = nn.DataParallel(fcrn_encode)
fcrn_encode = fcrn_encode.to(device)

if opt.load_model == 'True':
    fcrn_encode.load_state_dict(torch.load('./model/fcrn_encode_{}_link.pkl'.format(opt.alpha)))

fcrn_decode = Fcrn_decode(opt)
fcrn_decode = nn.DataParallel(fcrn_decode)
fcrn_decode = fcrn_decode.to(device)
if opt.load_model == 'True':
    fcrn_decode.load_state_dict(torch.load('./model/fcrn_decode_{}_link.pkl'.format(opt.alpha)))

Gen = Generator(opt)
Gen = nn.DataParallel(Gen)
Gen = Gen.to(device)
if opt.load_model == 'True':
    Gen.load_state_dict(torch.load('./model/Gen_{}_link.pkl'.format(opt.alpha)))

Dis = Discriminator(opt)
Dis = nn.DataParallel(Dis)
Dis = Dis.to(device)
if opt.load_model == 'True':
    Dis.load_state_dict(torch.load('./model/Dis_{}_link.pkl'.format(opt.alpha)))

Dis_optimizer = optim.Adam(Dis.parameters(), lr=opt.lr_1)
Dis_scheduler = optim.lr_scheduler.StepLR(Dis_optimizer, step_size=800, gamma=0.5)
Fcrn_encode_optimizer = optim.Adam(fcrn_encode.parameters(), lr=opt.lr)
encode_scheduler = optim.lr_scheduler.StepLR(Fcrn_encode_optimizer, step_size=300, gamma=0.5)
Fcrn_decode_optimizer = optim.Adam(fcrn_decode.parameters(), lr=opt.lr)
decode_scheduler = optim.lr_scheduler.StepLR(Fcrn_decode_optimizer, step_size=300, gamma=0.5)
Gen_optimizer = optim.Adam(Gen.parameters(), lr=opt.lr_1)
Gen_scheduler = optim.lr_scheduler.StepLR(Gen_optimizer, step_size=800, gamma=0.5)


# 训练函数
def train(device, train_dataloader, epoch):
    fcrn_encode.train()
    fcrn_decode.train()
    #     Gen.train()
    for batch_idx, (road, road_label) in enumerate(train_dataloader):
        road, road_label = road.to(device), road_label.to(device)

        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        img_noise = torch.cat((road, z), dim=1)
        fake_feature, n1, n2, n3 = Gen(img_noise)
        feature, x2, x3, x4 = fcrn_encode(road, n1, n2, n3)

        Dis_optimizer.zero_grad()
        d_real = Dis(feature.detach())
        d_loss_real = loss(d_real, 0.9 * torch.ones_like(d_real))
        d_fake = Dis((1 - opt.alpha) * feature.detach() + opt.alpha * fake_feature.detach())
        d_loss_fake = loss(d_fake, 0.1 + torch.zeros_like(d_fake))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        Dis_optimizer.step()

        Gen_optimizer.zero_grad()
        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        img_noise = torch.cat((road, z), dim=1)
        fake_feature, n1, n2, n3 = Gen(img_noise)
        detect_noise = fcrn_decode((1 - opt.alpha) * feature.detach() + opt.alpha * fake_feature, x2, x3, x4)
        d_fake = Dis((1 - opt.alpha) * feature.detach() + opt.alpha * fake_feature)
        g_loss = loss(d_fake, 0.9 * torch.ones_like(d_fake))
        g_loss -= loss(detect_noise, road_label)
        g_loss.backward()
        Gen_optimizer.step()

        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        img_noise = torch.cat((road, z), dim=1)
        fake_feature, n1, n2, n3 = Gen(img_noise)
        # feature_img = fake_feature.detach().cpu()
        # feature_img = np.transpose(np.array(utils.make_grid(feature_img, nrow=IMG_CUT)), (1, 2, 0))
        feature, x2, x3, x4 = fcrn_encode(road, n1, n2, n3)
        # detect = fcrn_decode(0.9*feature + 0.1*fake_feature)
        detect = fcrn_decode(feature, x2, x3, x4)
        # detect_img = detect.detach().cpu()
        # detect_img = np.transpose(np.array(utils.make_grid(detect_img, nrow=IMG_CUT)), (1, 2, 0))
        # blur = cv2.GaussianBlur(detect_img*255, (3, 3), 0)
        # _, thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
        fcrn_loss = loss(detect, road_label)
        fcrn_loss += torch.mean(torch.abs(detect - road_label)) / (torch.mean(torch.abs(detect + road_label)) + 0.001)
        Fcrn_encode_optimizer.zero_grad()
        Fcrn_decode_optimizer.zero_grad()
        fcrn_loss.backward()
        Fcrn_encode_optimizer.step()
        Fcrn_decode_optimizer.step()

        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        img_noise = torch.cat((road, z), dim=1)
        fake_feature, n1, n2, n3 = Gen(img_noise)
        # ffp, _ = torch.split(fake_feature, [3, 6*opt.dim-3], dim=1)
        # fake_feature_np = ffp.detach().cpu()
        # fake_feature_np = np.transpose(np.array(utils.make_grid(fake_feature_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        feature, x2, x3, x4 = fcrn_encode(road, n1, n2, n3)
        # fp, _ = torch.split(feature, [3, 6*opt.dim-3], dim=1)
        # feature_np = fp.detach().cpu()
        # feature_np = np.transpose(np.array(utils.make_grid(feature_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))

        road_np = road.detach().cpu()
        road_np = np.transpose(np.array(utils.make_grid(road_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        road_label_np = road_label.detach().cpu()
        road_label_np = np.transpose(np.array(utils.make_grid(road_label_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        detect_noise = fcrn_decode((1 - opt.alpha) * feature + opt.alpha * fake_feature.detach(), x2, x3, x4)
        detect_noise_np = detect_noise.detach().cpu()
        detect_noise_np = np.transpose(np.array(utils.make_grid(detect_noise_np, nrow=IMG_CUT, padding=0)), (1, 2, 0))
        blur = cv2.GaussianBlur(detect_noise_np * 255, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        fcrn_loss1 = loss(detect_noise, road_label)
        fcrn_loss1 += torch.mean(torch.abs(detect_noise - road_label)) / (
                    torch.mean(torch.abs(detect_noise + road_label)) + 0.001)

        Fcrn_decode_optimizer.zero_grad()
        Fcrn_encode_optimizer.zero_grad()
        fcrn_loss1.backward()
        Fcrn_decode_optimizer.step()
        Fcrn_encode_optimizer.step()

        writer.add_scalar('g_loss', g_loss.data.item(), global_step=batch_idx)
        writer.add_scalar('d_loss', d_loss.data.item(), global_step=batch_idx)
        writer.add_scalar('Fcrn_loss', fcrn_loss1.data.item(), global_step=batch_idx)

        if batch_idx % 20 == 0:
            tqdm.write(
                '[{}/{}] [{}/{}] Loss_Dis: {:.6f} Loss_Gen: {:.6f} Loss_Fcrn_encode: {:.6f} Loss_Fcrn_decode: {:.6f}'
                .format(epoch, num_epochs, batch_idx, len(train_dataloader), d_loss.data.item(), g_loss.data.item(),
                        (fcrn_loss.data.item()) / 2, (fcrn_loss1.data.item()) / 2))
        if batch_idx % 300 == 0:
            mix = np.concatenate(((road_np + 1) * 255 / 2, road_label_np * 255, detect_noise_np * 255), axis=0)
            # feature_np = cv2.resize((feature_np + 1)*255/2, (opt.image_scale_w, opt.image_scale_h))
            # fake_feature_np = cv2.resize((fake_feature_np + 1)*255/2, (opt.image_scale_w, opt.image_scale_h))
            # mix1 = np.concatenate((feature_np, fake_feature_np), axis=0)
            cv2.imwrite("./results/dete{}_{}.png".format(epoch, batch_idx), mix)
            # cv2.imwrite('./results_fcrn_noise/feature{}_{}.png'.format(epoch, batch_idx), mix1)
# cv2.imwrite("./results/feature{}_{}.png".format(epoch, batch_idx), (feature_img + 1)*255/2)
# cv2.imwrite("./results9/label{}_{}.png".format(epoch, batch_idx), np.transpose(road_label.cpu().numpy(), (2, 0, 1))*255)

# 测试函数
def test(device, test_dataloader):
    fcrn_encode.eval()
    fcrn_decode.eval()
#     Gen.eval()
    for batch_idx, (road, road_label, img_name)in enumerate(test_dataloader):
        road, _ = road.to(device), road_label.to(device)
        # z = torch.randn(road.shape[0], 1, IMAGE_SCALE, IMAGE_SCALE, device=device)
        # img_noise = torch.cat((road, z), dim=1)
        # fake_feature = Gen(img_noise)
        feature, x2, x3, x4  = fcrn_encode(road)
        det_road = fcrn_decode(feature, x2, x3, x4)
        label = det_road.detach().cpu()
        label = np.transpose(np.array(utils.make_grid(label, padding=0, nrow=1)), (1, 2, 0))
        # blur = cv2.GaussianBlur(label*255, (5, 5), 0)
        _, thresh = cv2.threshold(label*255, 200, 255, cv2.THRESH_BINARY)
        cv2.imwrite('./test/lab_dete_AVD/{}.png'.format(int(img_name[0])), thresh)
        print('testing...')
        print('{}/{}'.format(batch_idx, len(test_dataloader)))
    print('Done!')

# 文件的读取与保存
def iou(path_img, path_lab, epoch):
    img_name = os.listdir(path_img)
    img_name.sort(key=lambda x:int(x[:-4]))
    print(img_name)
    iou_list = []
    for i in range(len(img_name)):
        det = img_name[i]
        det = cv2.imread(path_img + '/' + det, 0)
        lab = img_name[i]
        lab = cv2.imread(path_lab + '/' + lab[:-4] + '.png', 0)
        lab = cv2.resize(lab, (opt.image_scale_w, opt.image_scale_h))
        count0, count1, a, count2 = 0, 0, 0, 0
        for j in range(det.shape[0]):
            for k in range(det.shape[1]):
                if det[j][k] != 0 and lab[j][k] != 0:
                    count0 += 1
                elif det[j][k] == 0 and lab[j][k] != 0:
                    count1 += 1
                elif det[j][k] != 0 and lab[j][k] == 0:
                    count2 += 1
                #iou = (count1 + count2)/(det.shape[0] * det.shape[1])
                iou = count0/(count1 + count0 + count2 + 0.0001)
        iou_list.append(iou)
        print(img_name[i], ':', iou)
    print('mean_iou:', sum(iou_list)/len(iou_list))
    with open('./munich_iou.txt',"a") as f:
        f.write("model_num" + " " + str(epoch) + " " + 'mean_iou:' + str(sum(iou_list)/len(iou_list)) + '\n')


# 主函数
if __name__ == '__main__':
    if opt.mode == 'train':
        num_epochs = opt.num_epochs
        for epoch in tqdm(range(num_epochs)):
            train(device, train_dataloader, epoch)
            Dis_scheduler.step()
            Gen_scheduler.step()
            encode_scheduler.step()
            decode_scheduler.step()
            if epoch % 50 == 0:
                now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                torch.save(Dis.state_dict(), './model/Dis_{}' + now + 'munich.pkl'.format(opt.alpha))
                torch.save(Gen.state_dict(), './model/Gen_{}' + now + 'munich.pkl'.format(opt.alpha))
                torch.save(fcrn_decode.state_dict(), './model/fcrn_decode_{}' + now + 'munich.pkl'.format(opt.alpha))
                torch.save(fcrn_encode.state_dict(), './model/fcrn_encode_{}' + now + 'munich.pkl'.format(opt.alpha))
                print('testing...')
                test(device, test_dataloader)
                iou('./test/lab_dete_AVD', './munich/test/lab', epoch)

    if opt.mode == 'test':
        test(device, test_dataloader)
        iou('./test/lab_dete_AVD', './munich/test/lab', 'test')
