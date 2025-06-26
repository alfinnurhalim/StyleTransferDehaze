import torch.nn as nn
import torch.nn.functional as F
import torch

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def weighted_mse_loss_merge(input_mean, target_mean, input_std, target_std, keep_ratio=1):
    loss_mean = ((input_mean - target_mean) ** 2)
    sort_loss_mean,idx = torch.sort(loss_mean,dim=1)

    sort_loss_mean[:,int(sort_loss_mean.shape[1]*keep_ratio):] = 0

    loss_std = ((input_std - target_std) ** 2)
    loss_std[:,idx[:,int(idx.shape[1]*keep_ratio):]] = 0

    return sort_loss_mean.mean(),loss_std.mean()

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 31

        self.mse_loss = nn.MSELoss()
        
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_style_loss(self, input, target, keep_ratio=1):
        # print(f'{input.size()} {target.size()}')
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)

        loss_mean,loss_std = weighted_mse_loss_merge(input_mean,
                                                    target_mean,
                                                    input_std,
                                                    target_std, 
                                                    keep_ratio=keep_ratio)
        return loss_mean+loss_std
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        
        size1 = input.size()
        size2 = target.size()
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)

        normalized_feat1 = (input - input_mean.expand(size1)) / input_std.expand(size1)
        normalized_feat2 = (target - target_mean.expand(size2)) / target_std.expand(size2)

        return self.mse_loss(normalized_feat1, normalized_feat2)

    def gram_matrix(self, feat):
        B, C, H, W = feat.size()
        feat = feat.view(B, C, H * W)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (C * H * W)
        return gram

    def cat_tensor(self,img):
        feat = self.encode_with_intermediate(img)
        mean,std = calc_mean_std(feat[0])
        mean = mean.squeeze(2)
        mean = mean.squeeze(2)
        std = std.squeeze(2)
        std = std.squeeze(2)
        t = torch.cat([mean,std],dim=1)
        for i in range(1,len(feat)):
            mean,std = calc_mean_std(feat[i])
            mean = mean.squeeze(2)
            mean = mean.squeeze(2)
            std = std.squeeze(2)
            std = std.squeeze(2)
            t = torch.cat([t,mean,std],dim=1)
        return t

    def forward(self, content_images, style_images, stylized_images, keep_ratio=None):
        content_feats = self.encode_with_intermediate(content_images)
        style_feats = self.encode_with_intermediate(style_images)
        stylized_feats = self.encode_with_intermediate(stylized_images)

        # print(content_feats[-1].shape)
        # print(style_feats[-1].shape)
        # print(stylized_feats[-1].shape)

        # Structure Loss
        loss_r = 0
        # for i in range(2):
        loss_r += self.calc_content_loss(stylized_feats[-1], content_feats[-1])

        # Style Loss
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i],keep_ratio=keep_ratio)

        loss_c = torch.zeros_like(loss_r)
        loss_p = torch.zeros_like(loss_r)

        return loss_c, loss_s, loss_r, loss_p
