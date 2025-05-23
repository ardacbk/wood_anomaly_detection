from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from flask_cors import CORS
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Configuration ---
MODEL_DIR = 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 2.2

app = Flask(__name__)
CORS(app)

# --- EfficientAD model initialization (from original working implementation) ---
def load_efficientad_model():
    from efficientad.efficientad import predict as efficientad_predict
   
    STUDENT_WEIGHTS = os.path.join(MODEL_DIR, 'efficientad', 'student.pth')
    TEACHER_WEIGHTS = os.path.join(MODEL_DIR, 'efficientad', 'teacher.pth')
    AE_WEIGHTS = os.path.join(MODEL_DIR, 'efficientad', 'autoencoder.pth')
   
    import torch.serialization as _ser
    from torch.nn.modules.container import Sequential
    with _ser.safe_globals([Sequential]):
        teacher = torch.load(TEACHER_WEIGHTS, map_location=DEVICE, weights_only=False).to(DEVICE)
        student = torch.load(STUDENT_WEIGHTS, map_location=DEVICE, weights_only=False).to(DEVICE)
        ae = torch.load(AE_WEIGHTS, map_location=DEVICE, weights_only=False).to(DEVICE)

    teacher.eval()
    student.eval()
    ae.eval()
   
    return {
        'teacher': teacher,
        'student': student,
        'autoencoder': ae,
        'predict_fn': efficientad_predict
    }

# --- GLASS model initialization ---
def load_glass_model():
    try:
        from glass.backbones import load as load_backbone
        from glass.glass import GLASS
        from glass.model import PatchMaker
        
        # Model konfigürasyonu
        backbone_name = "wideresnet50"
        layers_to_extract_from = ["layer2", "layer3"]
        target_size = 288  # GLASS için orijinal boyut
        
        # Backbone yükleme
        backbone = load_backbone(backbone_name)
        
        # GLASS modelini oluştur
        glass_model = GLASS(DEVICE)
        glass_model.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=DEVICE,
            input_shape=(3, target_size, target_size),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392
        )
        
        # Checkpoint yükleme
        GLASS_WEIGHTS = os.path.join(MODEL_DIR, 'glass', 'ckpt_best.pth')
        checkpoint = torch.load(GLASS_WEIGHTS, map_location=DEVICE)
        
        # State_dict yükleme
        if 'discriminator' in checkpoint:
            glass_model.discriminator.load_state_dict(checkpoint['discriminator'])
            if "pre_projection" in checkpoint:
                glass_model.pre_projection.load_state_dict(checkpoint["pre_projection"])
        else:
            glass_model.load_state_dict(checkpoint, strict=False)
        
        glass_model.eval()
        
        # PatchMaker'ı manuel olarak ekleyin
        glass_model.patch_maker = PatchMaker(patchsize=3, stride=1)
        target_size = 288  # Eğitimde kullandığınız boyut
        
        return {
            'model': glass_model,
            'target_size': target_size,  # Boyut bilgisini ekliyoruz
            'threshold': 0.5  # Sınıflandırma eşiği
        }

    except Exception as e:
        print(f"Error loading GLASS model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# --- INP-Former model initialization (from new implementation) ---
def load_inpformer_model():
    try:
        from inpformer.inpformer import INPFormerPredictor
       
        # Try to find the model weights using the expected folder structure
        dataset_name = 'MVTec-AD'
        encoder_type = 'dinov2reg_vit_base_14'
        inp_resize = 448
        inp_crop = 392
        inp_number = 6
        model_base_folder_name = 'INP-Former-Multi-Class'

        specific_model_folder_name = f'{model_base_folder_name}_dataset={dataset_name}_Encoder={encoder_type}_Resize={inp_resize}_Crop={inp_crop}_INP_num={inp_number}'
        INP_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'inpformer', specific_model_folder_name, 'model.pth')

        # If the specific path doesn't exist, try a simpler path
        if not os.path.exists(INP_WEIGHTS_PATH):
            INP_WEIGHTS_PATH_ALT = os.path.join(MODEL_DIR, 'inpformer', 'model.pth')
            if os.path.exists(INP_WEIGHTS_PATH_ALT):
                INP_WEIGHTS_PATH = INP_WEIGHTS_PATH_ALT
            else:
                print(f"WARNING: INP-Former weights not found at: {INP_WEIGHTS_PATH} or {INP_WEIGHTS_PATH_ALT}")
                return None
        
        print(f"Loading INP-Former weights from: {INP_WEIGHTS_PATH}")
        predictor_instance = INPFormerPredictor(
            weights_path=INP_WEIGHTS_PATH,
            device=DEVICE,
            input_size_arg=inp_resize,
            crop_size_arg=inp_crop,
            encoder_name_arg=encoder_type,
            inp_num_arg=inp_number
        )
        return {
            'predictor_object': predictor_instance,
            'threshold': 0.5  # Initial threshold for INP-Former
        }
    except Exception as e:
        print(f"Error loading INP-Former model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Load models
models = {
    'efficientad': load_efficientad_model(),
    'glass': load_glass_model(),
    'inpformer': load_inpformer_model()
}

# Keep only successfully loaded models
active_models = {name: data for name, data in models.items() if data is not None}
if not active_models:
    print("WARNING: No models were successfully loaded!")

# Preprocessing transform for EfficientAD and GLASS
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def bytes_to_pil_image(image_bytes) -> Image.Image:
    """Convert raw image bytes to PIL Image"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Could not convert image to PIL Image: {e}")

def preprocess_image(image_bytes, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Apply preprocessing to image bytes and return PIL Image"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image (in BGR format as OpenCV uses BGR)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
    if img_bgr is None:
        raise ValueError("Invalid image format or corrupted image data")
   
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   
    # Apply Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    # Apply morphological operations to close small holes
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
   
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    processed_img = None
   
    if contours:
        try:
            # Find the largest contour (main object)
            max_contour = max(contours, key=cv2.contourArea)
           
            # Create a mask (initially black, same size as the grayscale image)
            mask = np.zeros_like(gray)
            # Fill the contour with white
            cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
           
            # Apply the mask to the original BGR image
            masked_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
           
            # Crop to contain only the object area
            x, y, w, h = cv2.boundingRect(max_contour)
           
            # Check dimensions before cropping
            if w > 0 and h > 0:
                cropped_masked_bgr = masked_bgr[y:y+h, x:x+w]
                # Resize to target size
                resized_bgr = cv2.resize(cropped_masked_bgr, target_size, interpolation=cv2.INTER_AREA)
                processed_img = resized_bgr
            else:
                # If cropping fails, use masked but uncropped image
                resized_bgr = cv2.resize(masked_bgr, target_size, interpolation=cv2.INTER_AREA)
                processed_img = resized_bgr
               
        except Exception as e:
            # In case of error, use the original image
            processed_img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    else:
        # If no contours found, resize the original image
        processed_img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
   
    # Convert from BGR to RGB for PIL compatibility
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
   
    # Convert to PIL Image
    pil_image = Image.fromarray(processed_img_rgb)
   
    return pil_image


def generate_heatmap(anomaly_map, original_image, is_anomaly, threshold_value, model_name=None):
    """Enhanced heatmap visualization using EfficientAD style for all models with improved anomaly visibility."""
    
    # Make sure anomaly_map is properly processed
    if not isinstance(anomaly_map, np.ndarray):
        if hasattr(anomaly_map, 'cpu') and hasattr(anomaly_map, 'numpy'): # PyTorch tensor
            anomaly_map = anomaly_map.squeeze().cpu().numpy()
        else: # Unexpected type
            print(f"Warning: anomaly_map has unexpected type: {type(anomaly_map)}. Using zero map.")
            if hasattr(original_image, 'size'):
                h, w = original_image.height, original_image.width
            else: # Fallback
                h, w = 256, 256
            anomaly_map = np.zeros((h // 4, w // 4))

    if anomaly_map.ndim == 0: # Scalar case
        anomaly_map = np.array([[anomaly_map]])

    # Handle multi-channel maps
    if len(anomaly_map.shape) > 2:
        if anomaly_map.shape[0] == 1 or anomaly_map.shape[-1] == 1:
             anomaly_map = np.squeeze(anomaly_map)
        elif anomaly_map.shape[0] == 3: # 3 channels case
            anomaly_map = np.mean(anomaly_map, axis=0)
        else: # Other cases
            anomaly_map = np.max(anomaly_map, axis=0)
    
    # Resize small maps
    if anomaly_map.shape[0] < 16 or anomaly_map.shape[1] < 16:
        target_h = max(64, original_image.height // 4 if hasattr(original_image, 'height') else 64)
        target_w = max(64, original_image.width // 4 if hasattr(original_image, 'width') else 64)
        if anomaly_map.size > 1 and anomaly_map.shape[0] > 0 and anomaly_map.shape[1] > 0:
            anomaly_map = cv2.resize(anomaly_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else: # Default map if corrupted
            anomaly_map = np.zeros((target_h, target_w))

    # Clean invalid values
    anomaly_map_max = np.max(anomaly_map[np.isfinite(anomaly_map)]) if np.any(np.isfinite(anomaly_map)) else 1.0
    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=anomaly_map_max, neginf=0.0)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    
    # Handle original image
    if not hasattr(original_image, 'resize') or not hasattr(original_image, 'convert'):
        print("Warning: original_image is not a PIL.Image. Using default image.")
        original_image = Image.new('RGB', (256,256), color='lightgray')

    # Display original image in grayscale
    resized_original = original_image.resize(tuple(map(int, anomaly_map.shape[::-1])), Image.LANCZOS)
    ax.imshow(resized_original.convert('L'), cmap='gray', aspect='auto')
    
    # Always use EfficientAD style regardless of model
    heatmap_cmap_anomaly = 'hot'  # EfficientAD style for all models
    heatmap_cmap_normal = 'Blues'  # Normal case
    
    if is_anomaly:
        # Apply logarithmic scaling to highlight low value anomalies
        log_map = np.log1p(anomaly_map)
        
        # Use safe threshold value
        safe_threshold_value = max(threshold_value, 1e-6)
        log_threshold = np.log1p(safe_threshold_value)

        # EfficientAD visualization parameters for all models
        alpha_clip_min = 0.30
        alpha_clip_max = 0.80
        
        # GLASS model specific adjustment for heatmap visualization
        if model_name == 'glass':
            # Increase sensitivity for GLASS model by increasing the normalization factors
            # This will make the heatmap visualization show only stronger anomalies
            alpha_norm_low_factor = 0.9  # Higher value to filter out weaker signals (was 0.7)
            alpha_norm_high_factor = 1.5  # Higher value to highlight stronger signals (was 1.3)
        else:
            # Regular values for other models
            alpha_norm_low_factor = 0.7
            alpha_norm_high_factor = 1.3
        
        current_vmin = log_threshold * alpha_norm_low_factor
        # Set vmax based on log_map values or threshold
        current_vmax = max(log_threshold * alpha_norm_high_factor, np.percentile(log_map[log_map > current_vmin], 98) if np.any(log_map > current_vmin) else log_threshold * alpha_norm_high_factor)

        # Calculate denominator for alpha mask
        denominator = (log_threshold * alpha_norm_high_factor) - (log_threshold * alpha_norm_low_factor)
        if abs(denominator) < 1e-5:
            denominator = 1e-5 if denominator >= 0 else -1e-5
        
        # Create alpha mask with smooth transition
        alpha_mask = np.clip(
            (log_map - (log_threshold * alpha_norm_low_factor)) / denominator,
            0,
            1.0
        )
        alpha_mask = alpha_clip_min + alpha_mask * (alpha_clip_max - alpha_clip_min)
        alpha_mask[log_map < (log_threshold * alpha_norm_low_factor)] = 0

        # GLASS model specific adjustment to reduce noise in heatmap
        if model_name == 'glass':
            # Apply additional mask to filter out low intensity signals
            noise_filter_threshold = log_threshold * 0.95  # Increased from original
            alpha_mask[log_map < noise_filter_threshold] = 0

        # Ensure vmin and vmax are valid
        if current_vmax <= current_vmin:
            current_vmax = current_vmin + 1e-5

        # Render the heatmap
        heatmap_obj = ax.imshow(
            log_map,
            cmap=heatmap_cmap_anomaly,
            alpha=alpha_mask,
            vmin=current_vmin, 
            vmax=current_vmax,
            aspect='auto'
        )
        
        title_text = "ANOMALI TESPİT EDİLDİ"
        title_color = '#FF4500' # Orange-red
    
    else: # Normal case
        normal_display_vmax = threshold_value * 0.8
        normal_alpha = 0.3 if np.max(anomaly_map) > threshold_value * 0.1 else 0.1

        heatmap_obj = ax.imshow(
            anomaly_map, 
            cmap=heatmap_cmap_normal, 
            alpha=normal_alpha,    
            vmin=0,
            vmax=max(normal_display_vmax, 1e-5),
            aspect='auto'
        )
        title_text = "NORMAL (Anomali Yok)"
        title_color = '#2E8B57' # Sea green

    # Colorbar
    cbar = plt.colorbar(heatmap_obj, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Add threshold marker on colorbar
    if log_threshold > 1e-5 if is_anomaly else threshold_value > 1e-5:
        display_threshold_on_cbar = log_threshold if is_anomaly else threshold_value
        cbar_vmin, cbar_vmax = heatmap_obj.get_clim()

        if cbar_vmin <= display_threshold_on_cbar <= cbar_vmax:
            cbar.ax.axhline(y=display_threshold_on_cbar, 
                            color='#00FF00', linewidth=1.5, linestyle='--')
            cbar.ax.text(0.5, display_threshold_on_cbar, 
                        ' Eşik', transform=cbar.ax.get_yaxis_transform(),
                        va='center', ha='left', color='lime', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Set title
    ax.set_title(title_text, 
                 color=title_color, 
                 fontsize=15,
                 fontweight='bold',
                 pad=15,
                 bbox=dict(facecolor='black', alpha=0.7, pad=5, edgecolor='gray'))
    ax.axis('off')
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    
    return buf

@app.route('/api/models', methods=['GET'])
def get_available_models():
    # Return list of available models
    default_model = 'efficientad' if 'efficientad' in active_models else list(active_models.keys())[0] if active_models else None
    return jsonify({
        'available_models': list(active_models.keys()),
        'default_model': default_model
    })

@app.route('/api/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    model_name = request.form.get('model', 'efficientad')
    if model_name not in active_models:
        return jsonify({'error': f'Model {model_name} not available'}), 400

    try:
        model_data = active_models[model_name]
        
        if model_name == 'efficientad':
            default_thresh = 2.05  
        elif model_name == 'inpformer':
            default_thresh = 0.4
        else: # Glass and others
            default_thresh = model_data.get('threshold', DEFAULT_THRESHOLD * 0.7) 
            
        threshold = float(request.form.get('threshold', default_thresh))
        
        img_bytes = file.read()
        
        # Variables to store final score, anomaly state and raw anomaly map
        final_score_to_report = 0.0
        final_is_anomaly_to_report = False
        anomaly_map_for_visualization_np = None 
        original_pil_image_for_heatmap = None

        with torch.no_grad():
            if model_name == 'efficientad':
                preprocessed_image = preprocess_image(img_bytes, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                input_tensor = preprocess(preprocessed_image).unsqueeze(0).to(DEVICE)
                
                teacher_mean = torch.tensor(0.).to(DEVICE)
                teacher_std = torch.tensor(1.).to(DEVICE)
                combined_map_tensor, student_map, teacher_map = model_data['predict_fn'](
                    image=input_tensor,
                    teacher=model_data['teacher'],
                    student=model_data['student'],
                    autoencoder=model_data['autoencoder'],
                    teacher_mean=teacher_mean,
                    teacher_std=teacher_std
                )
                anomaly_map_for_visualization_np = combined_map_tensor.squeeze().cpu().numpy()
                original_pil_image_for_heatmap = preprocessed_image
                
                # Calculate score and anomaly status
                if anomaly_map_for_visualization_np.size == 0:
                    final_score_to_report = 0.0
                    final_is_anomaly_to_report = False
                else:
                    final_score_to_report = float(np.max(anomaly_map_for_visualization_np))
                    final_is_anomaly_to_report = final_score_to_report > threshold
            
            elif model_name == 'glass':
                pre_img = preprocess_image(img_bytes, target_size=(model_data['target_size'], model_data['target_size']))
                inp = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])(pre_img).unsqueeze(0).to(DEVICE)

                # Process with GLASS model
                _, list_of_raw_pixel_maps = model_data['model']._predict(inp)
                raw_map_from_glass_model = list_of_raw_pixel_maps[0]
                
                # Calculate score and anomaly status
                if raw_map_from_glass_model.size == 0:
                    final_score_to_report = 0.0
                    final_is_anomaly_to_report = False
                else:
                    final_score_to_report = float(np.max(raw_map_from_glass_model))
                    final_is_anomaly_to_report = final_score_to_report > threshold

                anomaly_map_for_visualization_np = np.squeeze(raw_map_from_glass_model)
                original_pil_image_for_heatmap = pre_img
                
            elif model_name == 'inpformer':
                # Apply the same preprocessing as other models
                preprocessed_image = preprocess_image(img_bytes, target_size=(448, 448))  # INP-Former's typical input size
                
                inpformer_predictor = model_data.get('predictor_object')
                if not inpformer_predictor:
                    return jsonify({'error': 'INP-Former predictor not loaded correctly.'}), 500
                
                # Use the preprocessed image instead of raw bytes
                anomaly_map_tensor = inpformer_predictor.predict(preprocessed_image)
                anomaly_map_for_visualization_np = anomaly_map_tensor.squeeze().cpu().numpy()
                original_pil_image_for_heatmap = preprocessed_image

                # Calculate score and anomaly status
                if anomaly_map_for_visualization_np.size == 0:
                    final_score_to_report = 0.0
                    final_is_anomaly_to_report = False
                else:
                    final_score_to_report = float(np.max(anomaly_map_for_visualization_np))
                    final_is_anomaly_to_report = final_score_to_report > threshold

        # Prepare heatmap visualization data
        is_anomaly_for_heatmap_display = final_is_anomaly_to_report
        map_to_use_in_heatmap = anomaly_map_for_visualization_np.copy()

        # Create default map if needed
        if map_to_use_in_heatmap is None or np.isnan(map_to_use_in_heatmap).all() or \
        (map_to_use_in_heatmap.size > 0 and np.max(map_to_use_in_heatmap) <= 1e-9 and np.min(map_to_use_in_heatmap) >= -1e-9) or \
        (map_to_use_in_heatmap.size == 0):
            
            ref_shape_h, ref_shape_w = IMAGE_SIZE // 4, IMAGE_SIZE // 4 
            if original_pil_image_for_heatmap is not None:
                if hasattr(original_pil_image_for_heatmap, 'size'):
                    ref_shape_h, ref_shape_w = original_pil_image_for_heatmap.height //4, original_pil_image_for_heatmap.width //4
                elif hasattr(original_pil_image_for_heatmap, 'shape'):
                    if original_pil_image_for_heatmap.ndim >=2:
                        ref_shape_h, ref_shape_w = original_pil_image_for_heatmap.shape[0] //4, original_pil_image_for_heatmap.shape[1] //4
            
            if not hasattr(map_to_use_in_heatmap, 'shape') or map_to_use_in_heatmap.ndim == 0 or map_to_use_in_heatmap.size == 0:
                map_to_use_in_heatmap = np.zeros((ref_shape_h, ref_shape_w)) + (threshold * 0.1 if threshold > 0 else 0.01)
            else:
                map_to_use_in_heatmap = np.zeros_like(map_to_use_in_heatmap) + (threshold * 0.1 if threshold > 0 else 0.01)
            is_anomaly_for_heatmap_display = False
        
        # Generate heatmap using EfficientAD style for all models
        heatmap_buffer = generate_heatmap(
            anomaly_map=map_to_use_in_heatmap,
            original_image=original_pil_image_for_heatmap,
            is_anomaly=is_anomaly_for_heatmap_display,
            threshold_value=threshold,
            model_name=model_name  # Always use EfficientAD style
        )
        
        heatmap_b64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
        
        original_buffer = io.BytesIO()
        if original_pil_image_for_heatmap is None:
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE)).save(original_buffer, format='PNG')
        else:
            original_pil_image_for_heatmap.save(original_buffer, format='PNG')
        original_buffer.seek(0)
        original_b64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'score': final_score_to_report,
            'threshold': threshold,
            'is_anomaly': final_is_anomaly_to_report,
            'status': 'ANOMALI TESPİT EDİLDİ!' if final_is_anomaly_to_report else 'NORMAL',
            'heatmap': heatmap_b64,
            'original': original_b64,
            'model': model_name
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)