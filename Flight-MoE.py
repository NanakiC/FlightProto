
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *
from ..models.layers.diffusion import *
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [N] (time steps)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x[:, None] -> [N, 1], emb[None, :] -> [1, D/2]
        emb = x[:, None] * emb[None, :] 
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
def generate_causal_mask(seq_len, device):
    

    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask.to(device)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        # Self-Attention 
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_linear = nn.Linear(d_model, d_model)
        # Cross-Attention
        # self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_linear = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, mask=None ,is_self_only=False):
        """
        x: [Batch, Seq_Len, D_Model] (Decoder Input)
        cross: [Batch, Seq_Len, D_Model] (Encoder Output)
        mask: [Seq_Len, Seq_Len] (Self-Only Mask)
        """
        # 1. Self-Attention with Self-Only Mask
        # src, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        src = self.self_attn_linear(x)
        x = self.norm1(x + self.dropout(src))

        # 2. Cross-Attention with Self-Only Mask (点对点恢复)
        # src, _ = self.cross_attn(x, cross, cross, attn_mask=mask, need_weights=False)
        src = self.cross_attn_linear(cross)
        x = self.norm2(x + self.dropout(src))

        # 3. Feed Forward
        src = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout(src))
        return x

class ProjectorDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_layers=1, dropout=0.1, activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, cross, mask=None):
        for layer in self.layers:
            x = layer(x, cross)
        return x
            
# Cell
class Flight_MoE(nn.Module):
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "pretrain", individual = False,representative_trajs=None,
                 y_range:Optional[tuple]=None, verbose:bool=False,head_wise_gate:bool=True, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction']
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, head_wise_gate=head_wise_gate,**kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_patch = num_patch
        if representative_trajs is not None:
            self.representative_trajs=torch.from_numpy(representative_trajs)
            self.num_patterns=self.representative_trajs.shape[0]
            self.mixture_projection = nn.Linear(patch_len, d_model)
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

        if head_type == "pretrain":
            self.decoder = ProjectorDecoder(d_model=d_model, d_ff=d_ff, 
                                            n_layers=1, dropout=dropout, activation=act)
            self.time_proj = nn.Sequential(
                SinusoidalPosEmb(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            self.diffusion = Diffusion(time_steps=1000, scheduler="cosine")
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
            # self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, num_patch*patch_len, head_dropout)
            
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)

            self.acc_embedding = nn.Linear(patch_len, d_model)
            self.acc_fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.acc_projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def compute_simple_spd(self, X_input, representative_trajs):
        """
        简化 SPD / 平均 L2
        X_input: [B, N, F, S] - batch, num_patches, features, patch_length
        representative_trajs: [P, S, F] - 代表轨迹的patch表示
        输出: [B, P] 每条输入轨迹到每条代表轨迹的距离
        """
        
        # 调整representative_trajs的维度以匹配X_input
        # representative_trajs: [P, S, F] -> [P, F, S] -> [1, 1, P, F, S]
        rep_trajs = representative_trajs.permute(0, 2, 1)  # [P, F, S]
        
        # X_input: [B, N, F, S] -> [B, N, 1, F, S]
        # rep_trajs: [P, F, S] -> [1, 1, P, F, S]
        diff = X_input[:, :, None, :, :] - rep_trajs[None, None, :, :, :]  # [B, N, P, F, S]
        
        # 对特征维度F求norm（保持与原始版本一致）
        dists = torch.norm(diff, dim=-2)  # [B, N, P, S] - 对特征维度F求norm
        
        # 然后对patch长度维度S和patch个数维度N取平均
        dists = dists.mean(dim=-1)  # [B, N ,P] - 对S和N取平均
        
        return dists.to(torch.float32)
        
    
    def forward(self, z ,trans_norm_z,x_vel=None, patch_indices=None):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        trans_norm_z: tensor [bs x num_patch x n_vars x patch_len]
        x_vel: tensor [bs x num_patch x n_vars x patch_len]
        """
        

        if self.head_type == "pretrain":
            backbone_feat, enc_emb = self.backbone(z, patch_indices=patch_indices, is_pretrain=True)# backbone_feat: [bs x nvars x d_model x num_patch] #enc_emb:[bs * nvars x num_patch x d_model]
            bs, actual_num_patch, n_vars, patch_len = z.shape
            #[B*nvars, num_patch, patch_len]
            z_flat = z.permute(0, 2, 1, 3).reshape(bs * n_vars, actual_num_patch, patch_len)
            z_noisy_flat, noise, t = self.diffusion(z_flat)
            t_flat = t.view(-1)
            t_emb_flat = self.time_proj(t_flat)
            # 4. Decoder Input (Linear) 
            t_emb = t_emb_flat.view(bs * n_vars, actual_num_patch, -1)

            # 还原形状: [B, N, V, L]
            z_noisy = z_noisy_flat.reshape(bs, n_vars, actual_num_patch, patch_len).permute(0, 2, 1, 3)
            # dec_input: [bs * n_vars, num_patch, d_model]
            dec_input = self.backbone.embed(z_noisy, patch_indices)
            dec_input = dec_input + t_emb
            # Decoder Forward q=dec_input[bs * nvars x num_patch x d_model] k,v=context  [bs*nvars, num_patch, d_model]
            #[bs * n_vars, num_patch, d_model]
            context = backbone_feat.permute(0, 1, 3, 2).reshape(-1, actual_num_patch, self.d_model)
            

            # 5. Decoder Forward (Linear)
            dec_out = self.decoder(
                dec_input, 
                context, 
                mask=None  
            )
            dec_out = dec_out.view(z.shape[0], self.n_vars, actual_num_patch, self.d_model).permute(0, 1, 3, 2)
            pre_output = self.head(dec_out) # [bs x num_patch x nvars x patch_len]
            return pre_output
        else:
            backbone_feat, enc_emb = self.backbone(z, patch_indices=patch_indices, is_pretrain=False)
            if x_vel is not None:
                # [bs x num_patch x n_vars x patch_len] -> [bs x num_patch x n_vars x d]
                acc_feat = self.acc_embedding(x_vel)
                # 非线性增强
                acc_feat = self.acc_projector(acc_feat)
                #[B, N, V, D] -> [B, V, D, N] (为了匹配 backbone_feat)
                acc_feat = acc_feat.permute(0, 2, 3, 1)

                # 拼接: [B, V, 2D, N]
                combined_acc = torch.cat([backbone_feat, acc_feat], dim=2)
                gate_input = combined_acc.permute(0, 1, 3, 2)
                gate_acc = self.acc_fusion_gate(gate_input) # -> [B, V, N, D]
                gate_acc = gate_acc.permute(0, 1, 3, 2)     # -> [B, V, D, N]

                backbone_feat = backbone_feat + gate_acc * acc_feat
            if hasattr(self, 'representative_trajs'):
                # --- MoE 计算 (原始空间) ---
                representative_trajs = self.representative_trajs.to(device=z.device, dtype=z.dtype)
                
                # 计算距离 (用归一化后的原始数据 trans_norm_z)
                dists = self.compute_simple_spd(trans_norm_z, representative_trajs) # [B, N, P]
                weights = F.softmax(-dists, dim=-1) # [B, N, P]
                
                # 计算 Mixture (原始轨迹) -> [B, N, F, S] (S=patch_len)
                mixture = torch.einsum("bnp,psf->bnfs", weights, representative_trajs)
                
                # --- 投影与对齐 (关键步骤) ---
                # mixture: [B, N, F, S] -> 投影 -> [B, N, F, D] (D=d_model)
                mixture_feat = self.mixture_projection(mixture)
                
                # 调整 mixture_feat 形状以匹配 backbone_feat: 
                # [B, N, F, D] -> [B, F, D, N] (Batch, nvars, d_model, num_patch)
                mixture_feat = mixture_feat.permute(0, 2, 3, 1)
                
                # --- 特征融合 ---
                # 在 Channel 维度 (dim=2, 即 d_model) 拼接
                # backbone_feat: [B, F, D, N]
                # mixture_feat:  [B, F, D, N]
                combined = torch.cat([backbone_feat, mixture_feat], dim=2) 
                
                # 为了过 Linear 层 (fusion_gate)，我们需要把 D 维度放到最后
                # combined: [B, F, D*2, N] -> permute -> [B, F, N, D*2]
                combined_perm = combined.permute(0, 1, 3, 2)
                
                # 计算门控: [B, F, N, D]
                gate = self.fusion_gate(combined_perm)
                
                # 门控变回: [B, F, D, N]
                gate = gate.permute(0, 1, 3, 2)
                
                # 执行加权融合
                z_fused = (1 - gate) * backbone_feat + gate * mixture_feat
            else:
                z_fused = backbone_feat
            fine_output = self.head(z_fused)
            return fine_output


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, head_wise_gate=True,**kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn,head_wise_gate=head_wise_gate)
    def embed(self, x, patch_indices=None):
       
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        
        # Positional Encoding
        if patch_indices is not None:
            if patch_indices.dim() == 2: 
                    patch_indices = patch_indices.unsqueeze(1).repeat(1, n_vars, 1).reshape(bs * n_vars, num_patch)
            elif patch_indices.dim() == 3:
                    patch_indices = patch_indices.reshape(bs * n_vars, num_patch)

            W_pos_expanded = self.W_pos.unsqueeze(0).expand(bs * n_vars, -1, -1)
            indices_expanded = patch_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
            selected_W_pos = torch.gather(W_pos_expanded, 1, indices_expanded)
            u = self.dropout(u + selected_W_pos)
        else:
            u = self.dropout(u + self.W_pos)
            
        return u
    def forward(self, x, patch_indices=None, is_pretrain=True) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        _,num_patch,n_vars,_=x.shape
        u = self.embed(x, patch_indices)
        if is_pretrain:
            sos_expanded = self.sos_token.expand(u.shape[0], -1, -1)
            u = torch.cat([sos_expanded, u], dim=1)[:, :-1, :]


        attn_mask = generate_causal_mask(u.shape[1], device=u.device)
        # Encoder
        z = self.encoder(u,attn_mask=attn_mask)                                                     # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z, u
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,head_wise_gate=True):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn,head_wise_gate=head_wise_gate) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor,attn_mask:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores,attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False,head_wise_gate=True):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        if head_wise_gate:
 
            self.self_attn = GatedMultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        else:

            self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None,attn_mask:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src,attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src



