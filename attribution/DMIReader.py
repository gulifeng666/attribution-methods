import torch.nn as nn
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.bottleneck.readout_bottleneck import ReadoutBottleneck
from attribution_bottleneck.utils.misc import resize, replace_layer, to_np
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import itertools
class DMIEstimator(AttributionMethod):

    def __init__(self, model, local_layer,global_layer,dim ):
        super().__init__()
        self.local_layer = local_layer
        self.global_layer = global_layer
        self.model = model
        self.mlp = nn.Sequential(nn.Linear(dim,1))
    def register(self):
        def local_forward_hook(m,input,output):
            self.local_feature = input
        def global_forward_hook(m,input,output):
            self.global_feature = output
            self.mi = self.calculate_mi(self.local_feature,self.global_feature)
        self.local_layer.register_forward_hook(local_forward_hook)
        self.global_layer.register_forward_hook(global_forward_hook)

    def calculate_mi(self,local_feature, global_feature):
    #local_feature:N*C*H*W
    #glocal_feature:N*D
        n = local_feature.shape[0]
        Nhw = local_feature.shape[0]*local_feature.shape[2]*local_feature.shape[3]
        hw =  local_feature.shape[2]*local_feature.shape[3]
        mask = [list(range(0+hw*i*n,hw+hw*i*n))for i in range(n)]
        mask = list(itertools.chain(*mask))
        local_feature = local_feature.permute(0,2,3,1)
        local_feature = local_feature.view(-1,local_feature.shape[-1])
        score = torch.cat([local_feature.repeat(1,global_feature.shape[0]).view(local_feature.shape[0]*global_feature.shape[0],-1) ,global_feature.repeat(local_feature.shape[0],1).view(local_feature.shape[0]*global_feature.shape[0],-1)],-1)
        mi = torch.log(torch.exp(score[mask])/(torch.exp(score).mean()))
    def detach(self):
        self.local_layer.remove()
        self.global_layer.remove()
    def few_shot_heatmap(self,model,x,iter,tag,n_support = 1):

        heatmaps = self.heatmap(x.reshape(-1, x.shape[2], x.shape[3], x.shape[4]), [])

        #          heatmaps[i,j] = torch.from_numpy(reader.heatmap(x[i,j].squeeze(0)))
        # summary.add_image(tag+'heatmapresult', heatmaps, iter, dataformats='HW')
        normal_x = (x - x.min()) / (x.max() - x.min())
        grid = make_grid(normal_x.view(-1, 3, 224, 224), nrow=x.shape[1])
        self.bn_layer.summary.add_image(tag + 'imresult', grid, iter, dataformats='CHW')
        figure = plt.figure(figsize=(4, 10))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.subplot(x.shape[0], x.shape[1], i * x.shape[1] + j + 1)
                plt.imshow(normal_x[i, j].cpu().permute(1, 2, 0))
                if (j == n_support):
                    plt.title('query', fontsize=10)
                else:
                    plt.title('support', fontsize=10)
                # heatmaps[i,j] = torch.from_numpy(reader.heatmap(x[i,j].squeeze(0)))
                plt.axis('off')
        plt.tight_layout()
        self.bn_layer.summary.add_figure(tag + 'imageresult', figure, iter)
        scores = model.set_forward(x)
        figure = plt.figure(figsize=(4, 10))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.subplot(x.shape[0], x.shape[1], i * x.shape[1] + j + 1)
                plt.imshow(heatmaps[i * x.shape[1] + j], cmap="jet")
                # plt.imshow(heatmaps[ i * x.shape[1] + j ], cmap="jet")
                if (j == n_support):
                    plt.title('score:' + str(np.around((nn.Softmax(0)(scores[i])).detach().cpu().numpy(), decimals=2)),
                              fontsize=5)
                else:
                    pass
                plt.axis('off')
        plt.tight_layout()
        self.bn_layer.summary.add_figure(tag + 'heatmapresult', figure, iter)
        grid = make_grid(torch.tensor(heatmaps).view(-1, 1, 224, 224), nrow=x.shape[1])
        self.bn_layer.summary.add_image(tag + 'hpresult', grid, iter, dataformats='CHW')

    # scores = self.set_forward(x)
    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):

        self.model.eval()
        self._inject_bottleneck()
        with torch.no_grad():
            self.model(input_t)
        self._remove_bottleneck()
        if(input_t.shape[0]>1):
            hmaps = np.zeros((input_t.shape[0],input_t.shape[2],input_t.shape[3]))
            for i in range(input_t.shape[0]):
              htensor = to_np(self.bn_layer.buffer_capacity[i])
              hmap = htensor.mean(0)
              hmap = resize(hmap, input_t.shape[2:])

              hmap = hmap - hmap.min()
              hmap = hmap/(max(hmap.max(),1e-5))
              hmaps[i] = hmap
            hmap = hmaps
        else:
            htensor = to_np(self.bn_layer.buffer_capacity)
            hmap = htensor.mean(axis=(0, 1))
            hmap = resize(hmap, input_t.shape[2:])
            hmap = hmap - hmap.min()
            hmap = hmap / (max(hmap.max(), 1e-5))

        return hmap
