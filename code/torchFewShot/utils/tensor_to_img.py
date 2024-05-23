import torch
import matplotlib.pyplot as plt
from torchvision import utils,transforms

#以Image的格式进行存储
def imageSavePIL(images,fileName,normalization=True,mean=0,std=1):
    image=utils.make_grid(images, 5)
    #是否原图进行了normalization
    if normalization:
        #交换之后，(H,W,C)
        image=image.permute(1,2,0)
        image=(image*torch.tensor(std)+torch.tensor(mean))
        #交换之后,(C,H,W)
        image=image.permute(2,0,1)
    #将tensor转化为Image格式
    image=transforms.ToPILImage()(image)
    #存储图片
    image.save(fileName)

#用plt库进行存储
def imageSavePLT(images,fileName,normalization=True,mean=0,std=1):
    image = utils.make_grid(images, 5)
    #交换维度之后 (H,W,C)
    image = image.permute(1,2,0)
    if normalization:
        image=(image*torch.tensor(std)+torch.tensor(mean)).numpy()
    #存储图片
    plt.imsave(fileName,image)
