# inpformer/inpformer.py

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torchvision import transforms
import torch.nn.functional as F
import math # get_gaussian_kernel için
from PIL import Image # <<< EKLENDİ

# Orijinal modelinizin dosyalarından importlar
from .models import vit_encoder
# 'models.uad' ve 'models.vision_transformer' modüllerinin Python path'inizde
# veya backend projenizin ana dizinine göre doğru bir şekilde erişilebilir olduğundan emin olun.
# Eğer bu modüller backend projenizin bir alt klasöründeyse (örn: 'original_model_files/models/uad')
# sys.path.append() kullanmanız veya projenizi Python paketi olarak yapılandırmanız gerekebilir.
# Şimdilik, bu importların çalıştığını varsayıyoruz.
from .models.uad import INP_Former
from .models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# utils.py'den alınan veya kopyalanan fonksiyonlar
# (cal_anomaly_maps_for_predictor ve get_gaussian_kernel_for_predictor önceki yanıttaki gibi kalacak)
def cal_anomaly_maps_for_predictor(fs_list, ft_list, out_size_hw_tuple):
    if not isinstance(out_size_hw_tuple, tuple):
        out_size_hw_tuple = (out_size_hw_tuple, out_size_hw_tuple)

    a_map_list = []
    for i in range(len(fs_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        if fs.ndim == 4 and ft.ndim == 4:
            a_map = 1 - F.cosine_similarity(fs, ft, dim=1)
            a_map = torch.unsqueeze(a_map, dim=1)
        else:
            raise ValueError(f"Beklenmeyen özellik tensör boyutu. fs: {fs.shape}, ft: {ft.shape}")
        a_map = F.interpolate(a_map, size=out_size_hw_tuple, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    if not a_map_list:
        raise ValueError("Anomali haritası listesi boş.")
    anomaly_map_combined = torch.cat(a_map_list, dim=1)
    final_anomaly_map = torch.mean(anomaly_map_combined, dim=1, keepdim=True)
    return final_anomaly_map

def get_gaussian_kernel_for_predictor(kernel_size=5, sigma=4, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel_val = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel_val = gaussian_kernel_val / torch.sum(gaussian_kernel_val)
    gaussian_kernel_val = gaussian_kernel_val.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel_val = gaussian_kernel_val.repeat(channels, 1, 1, 1)
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels, bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel_val
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

class INPFormerPredictor:
    def __init__(self, weights_path, device='cuda',
                 input_size_arg=448, crop_size_arg=392,
                 encoder_name_arg='dinov2reg_vit_base_14', inp_num_arg=6):
        self.device = device
        self.input_size_for_resize = input_size_arg
        self.crop_size_for_centercrop = crop_size_arg
        self.encoder_name = encoder_name_arg
        self.inp_num = inp_num_arg

        if 'small' in self.encoder_name:
            self.embed_dim, self.num_heads = 384, 6
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'base' in self.encoder_name:
            self.embed_dim, self.num_heads = 768, 12
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'large' in self.encoder_name:
            self.embed_dim, self.num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise ValueError(f"Desteklenmeyen encoder: {self.encoder_name}")

        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.model = self._build_model().to(self.device)
        self._load_weights(weights_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size_for_resize, self.input_size_for_resize)),
            transforms.ToTensor(),
            transforms.CenterCrop(self.crop_size_for_centercrop),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.gaussian_kernel = get_gaussian_kernel_for_predictor(kernel_size=5, sigma=4).to(self.device)
        self.final_map_target_size = 256

    def _build_model(self):
        encoder = vit_encoder.load(self.encoder_name)
        Bottleneck = nn.ModuleList([Mlp(self.embed_dim, self.embed_dim * 4, self.embed_dim, drop=0.)])
        INP = nn.ParameterList([nn.Parameter(torch.randn(self.inp_num, self.embed_dim))])
        INP_Extractor = nn.ModuleList([
            Aggregation_Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        ])
        num_decoder_blocks = 8
        INP_Guided_Decoder = nn.ModuleList([
            Prototype_Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(num_decoder_blocks)
        ])
        model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
                           decoder=INP_Guided_Decoder, target_layers=self.target_layers,
                           remove_class_token=True, fuse_layer_encoder=self.fuse_layer_encoder,
                           fuse_layer_decoder=self.fuse_layer_decoder, prototype_token=INP)
        return model

    def _load_weights(self, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"INP-Former weights '{weights_path}' yüklendi.")
        except Exception as e:
            print(f"INP-Former ağırlıkları yüklenirken hata: {e}")
            raise

    def predict(self, image_pil: Image.Image) -> torch.Tensor: # PIL Image alır, [H,W] CPU Tensor döndürür
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output_tuple = self.model(image_tensor)
            en_features = output_tuple[0]
            de_features = output_tuple[1]
            anomaly_map = cal_anomaly_maps_for_predictor(en_features, de_features,
                                                         out_size_hw_tuple=(self.crop_size_for_centercrop, self.crop_size_for_centercrop))
            if self.final_map_target_size is not None:
                anomaly_map = F.interpolate(anomaly_map,
                                            size=self.final_map_target_size,
                                            mode='bilinear',
                                            align_corners=False)
            anomaly_map = self.gaussian_kernel(anomaly_map)
        return anomaly_map.squeeze(0).squeeze(0).cpu() # CPU'ya alıp döndür