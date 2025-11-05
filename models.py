import torch
import torch.nn.functional as F
import torch.nn as nn


class Layer(nn.Module):

    def __init__(self,  out_features, nfea,beta,device):
        super(Layer, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)
        self.beta=beta
        self.device = device
    def forward(self, input, view,Lap=None):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        if Lap!=None:
            N = Lap.size(0)
            input3 = self.beta*torch.mm(Lap, input[:N,:])
            zeros_tail = input.new_zeros(input.size(0) - N, input.size(1))  # (N_mix, D)
            input3 = torch.cat([input3, zeros_tail], dim=0)
            output = input1 + input2-input3
        else:
            output = input1 + input2
        return output
class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=32):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list,weight):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class PUME(nn.Module):
    def __init__(self, nfeats, n_view,n_classes, layer_num, para, beta,device,hidden_dim=32,fusion_type='average'):
        super(PUME, self).__init__()
        self.n_classes = n_classes
        self.layer_num = layer_num
        self.device=device
        self.n_view=n_view
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)
        self.layers = nn.ModuleList([Layer(hidden_dim,feat,beta,device) for feat in nfeats])
        self.ZZ_init = nn.ModuleList([nn.Linear(feat,hidden_dim).to(device) for feat in nfeats])
        self.classifier=nn.Linear(hidden_dim, self.n_classes).to(device)
        self.fusionlayer = FusionLayer(n_view, fusion_type, self.n_classes, hidden_size=64)
        self.device=device
        self.weight= torch.zeros(self.n_view,1).to(device)

    def soft_threshold(self, u):
        return F.relu(u - self.theta) - F.relu(-1.0 * u - self.theta)

    def forward(self, features,Lap):
        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)
        for i in range(0, self.layer_num):
            Z = list()
            for view in range(0,self.n_view):
                if Lap!=None:
                    Z.append(self.soft_threshold(self.layers[view](output_z, features[view],Lap[view])))
                else:
                    Z.append(self.soft_threshold(self.layers[view](output_z, features[view])))
            output_z=self.fusionlayer(Z,self.weight)

        logit=self.classifier(output_z)
        return logit, Z
