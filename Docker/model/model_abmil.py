
import torch
import torch.nn as nn
import torch.nn.functional as F
 

def initialize_weights(module):
    for m in module.modules():
        if hasattr(m, 'weight') and m.weight is not None:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
        
class RNAEncoder(nn.Module):
    def __init__(self, input_dim=19360, hidden_dims=[2048, 512], output_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

"""
args:

"""
class ABMIL_Surv(nn.Module):
    def __init__(self, feat_type='uni', clinical_dim=27, rna_input_dim=19359):
        super().__init__()
        if feat_type == 'uni':
            size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            size = [1536, 768, 384]

        # MIL attention network for histo patches
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # RNA encoder
        self.rna_encoder = RNAEncoder(input_dim=rna_input_dim, output_dim=128)

        # Final classifier input: histo (size[1]) + clinical + rna(128)
        self.classifiers = nn.Linear(size[1] + clinical_dim + 128, 1)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.rna_encoder = self.rna_encoder.to(device)
        
    def forward(self, h, clinic, rna, attention_only=False):
        # h: (N_patches, patch_feat_dim)
        # clinic: (1, clinical_dim)
        # rna: (1, 19360)
        A, h_feat = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over patches N
        M = torch.mm(A, h_feat)  # 1 x size[1]

        # RNA feature vector
        rna_feat = self.rna_encoder(rna)  # (1, 128)

        # concatenate histo, clinical, and rna features
        combined = torch.cat([M, clinic, rna_feat], dim=1)  # (1, size[1]+clinical_dim+128)
        
        logits = self.classifiers(combined)  # (1,1)
        risk_score = logits.squeeze(1)  # no sigmoid in survival
        return risk_score, A_raw
    


class LowRankBilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, rank, out_dim):
        super().__init__()
        # Low-rank projections
        self.f1 = nn.Linear(dim1, rank)
        self.f2 = nn.Linear(dim2, rank)
        self.out = nn.Linear(rank, out_dim)

    def forward(self, x1, x2):
        # Project both to low-rank space
        p1 = self.f1(x1)  # (B, rank)
        p2 = self.f2(x2)  # (B, rank)
        # Element-wise interaction in low-rank space
        inter = p1 * p2
        return self.out(inter)  # (B, out_dim)


class ABMIL_Surv_PG(nn.Module):
    def __init__(self, feat_type='uni', clinical_dim=27, rna_input_dim=19359, rank=64):
        super().__init__()
        if feat_type == 'uni':
            size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            size = [1536, 768, 384]

        # Histo attention
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        fc.append(Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=1))
        self.attention_net = nn.Sequential(*fc)

        # RNA encoder
        self.rna_encoder = RNAEncoder(input_dim=rna_input_dim, output_dim=128)

        # Low-rank fusion modules
        self.fuse_hr = LowRankBilinearFusion(size[1], 128, rank, 64)       # histo × rna
        self.fuse_hc = LowRankBilinearFusion(size[1], clinical_dim, rank, 32)  # histo × clinic
        self.fuse_rc = LowRankBilinearFusion(128, clinical_dim, rank, 32)      # rna × clinic

        # Final classifier
        self.classifier = nn.Linear(64+32+32, 1)

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)
        self.rna_encoder = self.rna_encoder.to(device)
        self.fuse_hr = self.fuse_hr.to(device)
        self.fuse_hc = self.fuse_hc.to(device)
        self.fuse_rc = self.fuse_rc.to(device)

    def forward(self, h, clinic, rna, attention_only=False):
        A, h_feat = self.attention_net(h)  
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A

        A = F.softmax(A, dim=1)
        M = torch.mm(A, h_feat)  # histo representation

        rna_feat = self.rna_encoder(rna)

        # Low-rank fusions
        hr = self.fuse_hr(M, rna_feat)
        hc = self.fuse_hc(M, clinic)
        rc = self.fuse_rc(rna_feat, clinic)

        # Combine all fused interactions
        combined = torch.cat([hr, hc, rc], dim=1)

        logits = self.classifier(combined)  
        risk_score = logits.squeeze(1)
        return risk_score, A



class LowRankBilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, rank, out_dim):
        super().__init__()
        self.f1 = nn.Linear(dim1, rank)
        self.f2 = nn.Linear(dim2, rank)
        self.out = nn.Linear(rank, out_dim)

    def forward(self, x1, x2):
        p1 = self.f1(x1)
        p2 = self.f2(x2)
        inter = p1 * p2  # element-wise
        return self.out(inter)


class ABMIL_Surv_PG_Res(nn.Module):
    def __init__(self, feat_type='uni', clinical_dim=27, rna_input_dim=19359, rank=64):
        super().__init__()
        # Define histo attention network
        if feat_type == 'uni':
            size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            size = [1536, 768, 384]
            
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25), Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=1)]
        self.attention_net = nn.Sequential(*fc)

        # RNA encoder
        self.rna_encoder = RNAEncoder(input_dim=rna_input_dim, output_dim=128)

        # Low-rank fusion modules for pairs
        self.fuse_hr = LowRankBilinearFusion(size[1], 128, rank, 64)
        self.fuse_hc = LowRankBilinearFusion(size[1], clinical_dim, rank, 32)
        self.fuse_rc = LowRankBilinearFusion(128, clinical_dim, rank, 32)

        # Residual (direct concatenation) branch
        # Optional: you can project residual to smaller dim if needed
        self.residual_proj = nn.Linear(size[1] + 128 + clinical_dim, 128)

        # Final classifier
        # Total input dims: low-rank outputs + residual
        self.classifier = nn.Linear(64 + 32 + 32 + 128, 1)

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.rna_encoder = self.rna_encoder.to(device)
        self.residual_proj = self.residual_proj.to(device)
        self.classifier = self.classifier.to(device)
        self.fuse_hr = self.fuse_hr.to(device)
        self.fuse_hc = self.fuse_hc.to(device)
        self.fuse_rc = self.fuse_rc.to(device)

    def forward(self, h, clinic, rna, attention_only=False):
        # Histology processing
        A, h_feat = self.attention_net(h)  # attention gating + features
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A

        A = F.softmax(A, dim=1)
        M = torch.mm(A, h_feat)

        # RNA features
        rna_feat = self.rna_encoder(rna)

        # Low-rank interactions
        hr = self.fuse_hr(M, rna_feat)
        hc = self.fuse_hc(M, clinic)
        rc = self.fuse_rc(rna_feat, clinic)

        # Residual concatenation branch
        residual = torch.cat([M, rna_feat, clinic], dim=1)
        residual = F.relu(self.residual_proj(residual))

        # Combine all representations
        combined = torch.cat([hr, hc, rc, residual], dim=1)

        logits = self.classifier(combined)
        risk_score = logits.squeeze(1)
        return risk_score, A
