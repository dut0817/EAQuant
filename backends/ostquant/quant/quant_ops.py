import torch, torch.nn as nn, torch.nn.functional as F, math, fast_hadamard_transform
import utils.hadamard_utils as hadamard_utils
from utils import Rdtype
from einops import rearrange
from fast_hadamard_transform import hadamard_transform
from .quantizer import Quantizer


class H(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_hadamard_transform = False  
        self.online_full_had = False  
        self.online_partial_had = False  
        self.had_K = None
        self.K = 1
        self.down_dim = 1
        self.fp32_had = False

    def free_temporary(self):
        if hasattr(self, "temporary"):
            self.temporary = False
        if hasattr(self, "temp_weight"):
            self.temp_weight = None
        if hasattr(self, "temp_bias"):
            self.temp_bias = None

    def may_hadamard_transform(self, out):
        i_dtype = out.dtype
        if self.use_hadamard_transform:  
            out = hadamard_transform(
                out.float() if self.fp32_had else out, scale=1 / (math.sqrt(out.size(-1)))
            ).to(i_dtype)
        if self.online_full_had:  
            if self.down_dim != 1:
                shape = out.shape  
                out = out.reshape(shape[0], shape[1], -1, self.down_dim).transpose(2, 3)
                if self.fp32_had:
                    out = (
                        hadamard_utils.matmul_hadU_cuda(
                            (out.float() if self.fp32_had else out).contiguous(), self.had_K, self.K
                        )
                        .transpose(2, 3)
                        .reshape(shape)
                        .to(i_dtype)
                    )
                else:
                    out = (
                        hadamard_utils.matmul_hadU_cuda(
                            (out.float() if self.fp32_had else out).contiguous(), self.had_K, self.K
                        )
                        .transpose(2, 3)
                        .reshape(shape)
                    )
            else:
                if self.fp32_had:
                    out = hadamard_utils.matmul_hadU_cuda(
                        out.float(), self.had_K, self.K
                    ).to(i_dtype)
                else:
                    out = hadamard_utils.matmul_hadU_cuda(out, self.had_K, self.K).to(
                        i_dtype
                    )
        elif self.online_partial_had:
            init_shape = out.shape  
            if (
                self.K == 1
            ):
                if self.fp32_had:
                    out = (
                        fast_hadamard_transform.hadamard_transform(
                            out.float().permute(0, 2, 3, 1),
                            scale=1 / math.sqrt(out.shape[1]),
                        )
                        .permute(0, 3, 1, 2)
                        .to(i_dtype)
                    )
                else:
                    out = fast_hadamard_transform.hadamard_transform(
                        out.permute(0, 2, 3, 1), scale=1 / math.sqrt(out.shape[1])
                    ).permute(
                        0, 3, 1, 2
                    )  
            else:
                if self.fp32_had:
                    out = out.float()
                
                out = (
                    (
                        out.permute(0, 2, 3, 1)
                        @ self.had_K.to(dtype=out.dtype, device=out.device)
                    ).permute(0, 3, 1, 2)
                    / (math.sqrt(self.K))
                ).to(i_dtype)
                
                
                
        return out


class QuantLinear(nn.Linear, H):
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken"),
        act_quant_params=dict(bits=16, sym=True, dynamic_method="pertoken"),
        name=None,
        attn_instance=None,
    ):
        nn.Module.__init__(self)
        
        self.weight = org_module.weight.requires_grad_(False)
        if org_module.bias is not None:
            self.bias = org_module.bias.requires_grad_(False)
            
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.temporary = False

        self.weight_quantizer = Quantizer(**weight_quant_params)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.use_weight_quant = False

        self.name = name

        self.num_key_value_heads = (
            attn_instance.num_key_value_heads if attn_instance is not None else None
        )
        self.num_key_value_groups = (
            attn_instance.num_key_value_groups if attn_instance is not None else None
        )
        self.fast_recoder = False
    def forward(
        self,
        x,
        R_res=None,
        
        R_ov=None,
        S_up_down=None,
        S_qk=None,
        S_ov=None,
        S_up_gate=None,
        S_norm_qkv=None,
        S_norm_upgate=None,
    ):
        if self.temporary:
            ori_dtype = self.weight.dtype
            temp_weight = self.weight
            if self.bias is not None:
                temp_bias = self.bias

            
            if S_up_down is not None and self.name == "up":
                temp_weight = temp_weight.to(Rdtype) / (S_up_down.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_up_down.view(-1))
            if S_up_down is not None and self.name == "down":
                temp_weight = temp_weight.to(Rdtype) * (S_up_down.view(1, -1))

            
            if S_qk is not None and self.name == "k":
                temp_weight = temp_weight.to(Rdtype) * (S_qk.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) * (S_qk.view(-1))
            if S_qk is not None and self.name == "q":  
                if self.weight.shape[0] > S_qk.numel():
                    S_qk = S_qk.reshape(self.num_key_value_heads, -1)
                    n_head, d = S_qk.shape
                    S_qk = (
                        S_qk[:, None, :]
                        .expand(n_head, self.num_key_value_groups, d)
                        .reshape(-1)
                    )
                temp_weight = temp_weight.to(Rdtype) / (S_qk.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_qk.view(-1))

            
            if S_ov is not None and self.name == "v":
                temp_weight = temp_weight.to(Rdtype) / (S_ov.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_ov.view(-1))
            if S_ov is not None and self.name == "o":  
                if self.weight.shape[0] > S_ov.numel():
                    S_ov = S_ov.reshape(self.num_key_value_heads, -1)
                    n_head, d = S_ov.shape
                    S_ov = (
                        S_ov[:, None, :]
                        .expand(n_head, self.num_key_value_groups, d)
                        .reshape(-1)
                    )
                temp_weight = temp_weight.to(Rdtype) * (S_ov.view(-1, 1))

            
            if (
                S_up_gate is not None and self.name == "up"
            ):  
                temp_weight = temp_weight.to(Rdtype) / (S_up_gate.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) / (S_up_gate.view(-1))
            if S_up_gate is not None and self.name == "gate":
                temp_weight = temp_weight.to(Rdtype) * (S_up_gate.view(-1, 1))
                if self.bias is not None:
                    temp_bias = temp_bias.to(Rdtype) * (S_up_gate.view(-1))

            
            if R_res is not None and self.name in ["q", "k", "v", "up", "gate"]:
                temp_weight = temp_weight.to(Rdtype) @ R_res.to(Rdtype)
            elif R_res is not None and self.name in ["down", "o"]:
                temp_weight = R_res.T.to(Rdtype) @ temp_weight.to(Rdtype)
                if self.bias is not None:
                    temp_bias = R_res.T.to(Rdtype) @ temp_bias.to(Rdtype)

            
            if R_ov is not None and self.name == "v":
                R_ov = torch.stack(list(R_ov), dim=0)
                had_dim = R_ov.shape[-1]
                W_ = temp_weight.t()
                transposed_shape = W_.shape
                temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                
                temp = ((temp.unsqueeze(2).to(Rdtype)) @ (R_ov.to(Rdtype))).squeeze(2)
                temp_weight = temp.reshape(transposed_shape).t()
                if self.bias is not None:
                    temp_bias = (
                        temp_bias.to(Rdtype)
                        .reshape(self.num_key_value_heads, had_dim)
                        .unsqueeze(1)
                        @ R_ov.to(Rdtype)
                    ).squeeze(1).reshape(-1)
            if R_ov is not None and self.name == "o":
                R_ov = torch.stack(list(R_ov), dim=0)
                had_dim = R_ov.shape[-1]
                init_shape = temp_weight.shape
                temp = temp_weight.reshape(-1, init_shape[-1] // had_dim, had_dim)
                if self.num_key_value_groups != 1:
                    h, d1, d2 = R_ov.shape
                    repeated_R_ov = (
                        R_ov[:, None, :, :]
                        .expand(h, self.num_key_value_groups, d1, d2)
                        .reshape(-1, d1, d2)
                    )
                    
                    temp = (
                        (temp.unsqueeze(2).to(Rdtype)) @ (repeated_R_ov.to(Rdtype))
                    ).squeeze(2)
                else:
                    temp = ((temp.unsqueeze(2).to(Rdtype)) @ (R_ov.to(Rdtype))).squeeze(
                        2
                    )
                temp_weight = temp.reshape(init_shape)

            
            if S_norm_qkv is not None and self.name in ["q", "k", "v"]:
                temp_weight = temp_weight.to(Rdtype) * (
                    S_norm_qkv.view(1, -1)
                )  

            
            if S_norm_upgate is not None and self.name in ["up", "gate"]:
                temp_weight = temp_weight.to(Rdtype) * (S_norm_upgate.view(1, -1))

            
            if self.name == "head" and R_res is not None:
                temp_weight = temp_weight.to(Rdtype) @ R_res.to(Rdtype)
            weight = temp_weight.to(ori_dtype)
            if self.bias is not None:
                bias = temp_bias.to(ori_dtype)
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_weight_quant:
            weight = self.weight_quantizer(weight)

        out = F.linear(x, weight, bias)
        if self.use_act_quant:
            out = self.act_quantizer(out)

        return out


class QuantMatmul(H):
    def __init__(
        self,
        act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken"),
        matmul_func=torch.matmul,
        is_pvmat=False,  
        is_qkmat=False,
    ):
        super().__init__()
        self.matmul_func = matmul_func
        self.is_pvmat = is_pvmat
        self.is_qkmat = is_qkmat
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        if self.is_qkmat:
            out = torch.matmul(x1.float(), x2.float())
        else:
            out = torch.matmul(x1, x2)
        out = self.may_hadamard_transform(out)
        if self.is_pvmat:
            b, h, l, c = out.shape
            out = out.transpose(1, 2).contiguous().reshape(b, l, h * c)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantROPE(H):
    def __init__(
        self,
        rope_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertensor"),
        act_quant_params=dict(bits=8, sym=True, dynamic_method="pertensor"),
    ):
        super().__init__()
        self.rope_quantizer = Quantizer(**rope_quant_parmas)
        self.act_quantizer = Quantizer(**act_quant_params)

        self.use_act_quant = False
        self.use_weight_quant = False

    def forward(self, x, Wrope, pre_rope_Q=None, post_rope_Q=None):
        if pre_rope_Q is not None:
            Wrope = (pre_rope_Q.T.to(Rdtype)) @ Wrope.to(Rdtype)
        if post_rope_Q is not None:
            Wrope = Wrope.to(Rdtype) @ post_rope_Q.to(Rdtype)
        if self.use_weight_quant:
            Wrope = self.rope_quantizer(Wrope)
        out = torch.matmul(x, Wrope.to(x.dtype))
        out = self.may_hadamard_transform(out)
        if self.use_act_quant:
            b, l, h, d = out.shape
            out = out.reshape(b, l, -1)
            out = self.act_quantizer(out)
            out = out.reshape(b, l, h, d)

        return out


class QuantRMSNorm(H):
    weight: torch.Tensor

    def __init__(
        self,
        ori_norm,
        act_quant_params=dict(bits=8, symmetric=True, dynamic_method="perchannel"),
    ):
        super().__init__()
        self.eps = ori_norm.variance_epsilon
        self.act_quantizer = Quantizer(**act_quant_params)
        self.weight = ori_norm.weight.requires_grad_(False)
        self.bias = None

        self.temporary = False
        self.temp_weight = self.temp_bias = None
        self.use_act_quant = False

    def forward(self, hidden_states, S=None):
        i_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight
        if self.temporary and S is not None:  
            weight = weight / S

        out = (weight * hidden_states).to(i_dtype)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantAdd(H):
    def __init__(
        self, act_quant_parmas=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_parmas)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1 + x2
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSoftmax(H):
    def __init__(self, act_quant_params: dict = dict(), dim=-1):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_params)
        self.dim = dim

        self.use_act_quant = False

    def forward(self, attn_weights, attn_mask=None):
        i_dtype = attn_weights.dtype
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        out = F.softmax(attn_weights, dim=self.dim, dtype=torch.float32)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out.to(i_dtype)


class QuantMul(H):
    def __init__(
        self, act_quant_param=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_quantizer = Quantizer(**act_quant_param)
        self.use_act_quant = False

    def forward(self, x1, x2):
        out = x1 * x2
        
        
        out = self.may_hadamard_transform(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantSiLU(H):
    def __init__(
        self, act_quant_params=dict(bits=8, sym=True, dynamic_method="pertoken")
    ):
        super().__init__()
        self.act_func = F.silu
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False
        self.smooth = None
        self.temporary = False

    def forward(self, x, S_up_gate=None, **kwargs):
        if self.temporary:
            self.smooth = S_up_gate
        if self.smooth is None:
            out = F.silu(x)
        else:
            out = x * F.sigmoid(x / self.smooth.to(x.device))
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantEmbedding(nn.Embedding):
    def __init__(self, ori: nn.Embedding, act_quant_params=dict(bits=16)):
        super().__init__(
            num_embeddings=ori.num_embeddings,
            embedding_dim=ori.embedding_dim,
            padding_idx=ori.padding_idx,
            max_norm=ori.max_norm,
            norm_type=ori.norm_type,
            scale_grad_by_freq=ori.scale_grad_by_freq,
            sparse=ori.sparse,
            _weight=ori.weight,
            _freeze=True,
            device=ori.weight.device,
            dtype=ori.weight.dtype,
        )
        self.temporary = False
        del self.weight
        self.register_buffer("weight", ori.weight.data)
        self.act_quantizer = Quantizer(**act_quant_params)
        self.use_act_quant = False

    def forward(self, input: torch.Tensor, R_res=None) -> torch.Tensor:
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.temporary and R_res is not None:
            ori_dtype = out.dtype
            out = (
                (out.to(Rdtype))
                @ (R_res.to(dtype=Rdtype, device=out.device))
            ).to(ori_dtype)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out
