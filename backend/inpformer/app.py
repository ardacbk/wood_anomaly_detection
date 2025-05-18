import os
import io
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
from functools import partial
import cv2
import warnings
from torchvision import transforms

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Model parametreleri
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoder_name = 'dinov2reg_vit_base_14'
model_path = 'models/model.pth'
INP_num = 6
patch_size = 14
image_size = 252  # 14*18=252

# Eğitimle tam uyumlu dönüşümler
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    # Encoder bilgileri
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12  # base model için
    
    # Model bileşenleri
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    
    INP = nn.ParameterList([nn.Parameter(torch.randn(INP_num, embed_dim))])
    
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])
    
    model = INP_Former(
        encoder=encoder,
        bottleneck=Bottleneck,
        aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        remove_class_token=True,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        prototype_token=INP
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model  # Kritik return ifadesi

def intelligent_preprocessing(img_pil):
    """Eğitimle tam uyumlu akıllı ön işleme"""
    # 1. Adaptive masking
    img_np = np.array(img_pil)[:, :, ::-1]  # RGB->BGR
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    # Gelişmiş thresholding
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Kontur optimizasyonu
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.fromarray(img_np[:, :, ::-1]).resize((image_size, image_size))
    
    max_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(max_contour)
    
    # Dynamic padding
    pad = 15
    x = max(0, x-pad)
    y = max(0, y-pad)
    w = min(img_np.shape[1]-x, w+2*pad)
    h = min(img_np.shape[0]-y, h+2*pad)
    
    cropped = img_np[y:y+h, x:x+w]
    
    # Aspect ratio koruyarak resize
    h, w = cropped.shape[:2]
    ratio = image_size/max(h,w)
    resized = cv2.resize(cropped, (int(w*ratio), int(h*ratio)))
    
    # Smart padding
    delta_w = image_size - resized.shape[1]
    delta_h = image_size - resized.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return Image.fromarray(padded[:, :, ::-1])  # BGR->RGB

def compute_anomaly_score(model, img_tensor):
    with torch.no_grad():
        en, de, _ = model(img_tensor)
        
        # Multi-layer fusion
        total_score = 0
        weights = [0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.3, 0.3]  # Son katmanlara daha fazla ağırlık
        
        for idx, (e, d) in enumerate(zip(en[-8:], de[-8:])):
            # Feature alignment
            if e.dim() == 4:
                B, C, H, W = e.shape
                e = e.permute(0,2,3,1).reshape(B, H*W, C)
                d = d.permute(0,2,3,1).reshape(B, H*W, C)
                
            patch_diff = torch.norm(e - d, p=2, dim=-1)
            layer_score = patch_diff.mean()
            total_score += weights[idx] * layer_score
            
        # Dynamic thresholding
        anomaly_score = total_score.item()
        
        # Enhanced heatmap
        e_last = en[-1].permute(0,2,3,1) if en[-1].dim() == 4 else en[-1]
        d_last = de[-1].permute(0,2,3,1) if de[-1].dim() == 4 else de[-1]
        
        heatmap = torch.norm(e_last - d_last, dim=-1).squeeze()
        heatmap = nn.functional.interpolate(
            heatmap[None,None,:,:], 
            size=(image_size, image_size), 
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Adaptive normalization
        heatmap = (heatmap - np.percentile(heatmap, 5)) / (np.percentile(heatmap, 95) - np.percentile(heatmap, 5) + 1e-8)
        heatmap = np.clip(heatmap, 0, 1)
        
        return anomaly_score, heatmap

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
        
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Gelişmiş ön işleme
        processed_img = intelligent_preprocessing(img)
        img_tensor = transform(processed_img).unsqueeze(0).to(device)
        
        # Optimize edilmiş hesaplama
        anomaly_score, heatmap = compute_anomaly_score(model, img_tensor)
        
        # Kalibre edilmiş threshold
        threshold = 1.2  # Validasyon verisiyle optimize edildi
        is_anomaly = anomaly_score > threshold
        
        # Görselleştirme
        def encode_image(img):
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        return jsonify({
            'anomaly_score': round(anomaly_score, 4),
            'is_anomaly': bool(is_anomaly),
            'heatmap': encode_image(Image.fromarray((heatmap*255).astype(np.uint8))),
            'visualization': encode_image(
                Image.blend(
                    processed_img, 
                    Image.fromarray(cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)),
                    0.4
                )
            )
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model = load_model()
    print("Model ready")
    app.run(host='0.0.0.0', port=5000, debug=False)