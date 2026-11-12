import os
import time
import shutil
import io
import json
import zipfile
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from arnold_cat_map import (
    arnold_cat_map,
    logistic_map_scramble,
    logistic_map_descramble,
    inverse_arnold_cat_map,
    save_image_from_matrix
)
from image_to_rgb_matrix import image_to_rgb_matrix, crop_and_resize_to_square
from feature_extractor import extract_crypto_features, CLASS_LABELS, CLASS_SEVERITY, FEATURE_NAMES

UPLOAD_FOLDER = 'uploads'
ENCRYPTED_FOLDER = 'encrypted'
DECRYPTED_FOLDER = 'decrypted'
ARTIFACTS_FOLDER = 'ml_artifacts'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

MODEL_PATH = os.path.join(MODELS_FOLDER, 'crypto_classifier.joblib')
ML_PIPELINE = None

def get_ml_pipeline():
    global ML_PIPELINE
    if ML_PIPELINE is None and os.path.exists(MODEL_PATH):
        try:
            ML_PIPELINE = joblib.load(MODEL_PATH)
        except Exception as e:
            print("Warning: Failed to load ML pipeline:", e)
    return ML_PIPELINE

def predict_crypto_security(img_matrix):
    pipe = get_ml_pipeline()
    if pipe is None:
        return None
    try:
        feats = extract_crypto_features(img_matrix)
        feature_names = pipe.get("feature_names", FEATURE_NAMES)
        x_vec = np.array([[feats[f] for f in feature_names]])
        x_scaled = pipe["scaler"].transform(x_vec)
        pred_idx = int(pipe["classifier"].predict(x_scaled)[0])
        probs = pipe["classifier"].predict_proba(x_scaled)[0]
        conf = float(probs[pred_idx])
        
        prob_dict = {
            CLASS_LABELS[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }
        severity_info = CLASS_SEVERITY.get(pred_idx, {})
        status_txt = severity_info.get("status", "Unknown")
        risk_txt = severity_info.get("risk", "Unknown")
        
        return {
            "class_index": pred_idx,
            "class_id": pred_idx,
            "class_name": CLASS_LABELS.get(pred_idx, "Unknown"),
            "predicted_class": CLASS_LABELS.get(pred_idx, "Unknown"),
            "confidence": conf,
            "confidence_pct": round(conf * 100, 2),
            "status": status_txt,
            "risk": risk_txt,
            "risk_assessment": f"{status_txt}: {risk_txt}",
            "color": severity_info.get("color", "#06b6d4"),
            "probabilities": prob_dict,
            "model_name": pipe.get("model_name", "ML Classifier"),
            "features": {k: round(v, 4) for k, v in feats.items()}
        }
    except Exception as e:
        print("ML audit error:", e)
        return None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENCRYPTED_FOLDER'] = ENCRYPTED_FOLDER
app.config['DECRYPTED_FOLDER'] = DECRYPTED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload limit

for folder in [UPLOAD_FOLDER, ENCRYPTED_FOLDER, DECRYPTED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_security_metrics(img_matrix):
    """
    Computes Shannon Information Entropy and adjacent pixel correlation coefficients
    (horizontal and vertical) across the image.
    """
    try:
        flat = img_matrix.flatten()
        counts = np.bincount(flat, minlength=256)
        probs = counts[counts > 0] / len(flat)
        entropy = float(-np.sum(probs * np.log2(probs)))

        # Luminance for correlation
        gray = 0.2989 * img_matrix[:, :, 0] + 0.5870 * img_matrix[:, :, 1] + 0.1140 * img_matrix[:, :, 2]
        x_h, y_h = gray[:, :-1].flatten(), gray[:, 1:].flatten()
        x_v, y_v = gray[:-1, :].flatten(), gray[1:, :].flatten()

        samples = 3000
        if len(x_h) > samples:
            idx = np.random.choice(len(x_h), samples, replace=False)
            x_h, y_h = x_h[idx], y_h[idx]
            x_v, y_v = x_v[idx], y_v[idx]

        def corr(a, b):
            cov = np.cov(a, b)[0, 1]
            std_a, std_b = np.std(a), np.std(b)
            return float(cov / (std_a * std_b)) if std_a * std_b != 0 else 0.0

        return {
            'entropy': round(entropy, 4),
            'h_corr': round(corr(x_h, y_h), 4),
            'v_corr': round(corr(x_v, y_v), 4)
        }
    except Exception as e:
        return {'entropy': 0.0, 'h_corr': 0.0, 'v_corr': 0.0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample', methods=['POST'])
def load_sample_image():
    """
    Copies or creates a high-quality demonstration square image for 1-click testing.
    """
    sample_src = 'decrypted_image_gui.png'
    if not os.path.exists(sample_src):
        # Create a stylish test pattern if not present
        sz = 512
        arr = np.zeros((sz, sz, 3), dtype=np.uint8)
        x_idx, y_idx = np.meshgrid(np.arange(sz), np.arange(sz))
        arr[:, :, 0] = ((x_idx / sz) * 255).astype(np.uint8)
        arr[:, :, 1] = ((y_idx / sz) * 255).astype(np.uint8)
        arr[:, :, 2] = (((x_idx + y_idx) / (2 * sz)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(sample_src)

    filename = f"sample_{int(time.time())}.png"
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copyfile(sample_src, dest_path)

    mat = image_to_rgb_matrix(dest_path)
    metrics = compute_security_metrics(mat)
    ml_audit = predict_crypto_security(mat)

    return jsonify({
        'filename': filename,
        'width': mat.shape[1],
        'height': mat.shape[0],
        'size_kb': round(os.path.getsize(dest_path) / 1024, 1),
        'metrics': metrics,
        'ml_audit': ml_audit
    }), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PNG, JPG, BMP, or WebP.'}), 400

    auto_crop = request.form.get('auto_crop', 'true').lower() in ('true', '1', 'yes')
    target_dim = request.form.get('target_dim', '512')
    try:
        target_dim = int(target_dim) if target_dim and int(target_dim) > 0 else None
    except ValueError:
        target_dim = 512

    orig_name = secure_filename(file.filename)
    base_name = os.path.splitext(orig_name)[0]
    filename = f"upload_{int(time.time())}_{base_name}.png"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # Process image to ensure square NxN for Arnold Cat Map
    try:
        with Image.open(upload_path) as im:
            w, h = im.size
            is_square = (w == h)

        if not is_square and auto_crop:
            mat = crop_and_resize_to_square(upload_path, target_size=target_dim, save_path=upload_path)
        elif is_square and target_dim and w > target_dim:
            # Scale down large square images for blazing-fast encryption
            mat = crop_and_resize_to_square(upload_path, target_size=target_dim, save_path=upload_path)
        else:
            mat = image_to_rgb_matrix(upload_path)

        if mat.shape[0] != mat.shape[1]:
            return jsonify({'error': f'Image is not square ({mat.shape[1]}x{mat.shape[0]}). Enable "Auto-Crop to Square" to proceed.'}), 400

        metrics = compute_security_metrics(mat)
        ml_audit = predict_crypto_security(mat)
        return jsonify({
            'filename': filename,
            'width': mat.shape[1],
            'height': mat.shape[0],
            'size_kb': round(os.path.getsize(upload_path) / 1024, 1),
            'metrics': metrics,
            'ml_audit': ml_audit
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

@app.route('/encrypt', methods=['POST'])
def encrypt_image():
    start_time = time.time()
    data = request.json or {}
    filename = data.get('filename')
    x = data.get('x')
    r = data.get('r')
    iterations = data.get('iterations', 1)

    if not filename or x is None or r is None:
        return jsonify({'error': 'Missing required encryption parameters (filename, x, r)'}), 400

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(upload_path):
        return jsonify({'error': 'Uploaded source image not found on server'}), 400

    try:
        x = float(x)
        r = float(r)
        iterations = max(1, min(20, int(iterations)))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid numeric parameters for x, r, or iterations'}), 400

    if not (0.0 < x < 1.0):
        return jsonify({'error': 'Key x must be strictly between 0 and 1 (e.g. 0.5888)'}), 400
    if not (3.57 <= r <= 4.0):
        return jsonify({'error': 'Chaotic parameter r must be in chaotic region 3.57 <= r <= 4.0 (e.g. 3.98888)'}), 400

    try:
        rgb_matrix = image_to_rgb_matrix(upload_path)
        if rgb_matrix.shape[0] != rgb_matrix.shape[1]:
            return jsonify({'error': 'Image must be square (NxN) for Arnold Cat Map'}), 400

        # 1. Arnold Cat Map Spatial Scrambling
        scrambled_matrix = arnold_cat_map(rgb_matrix, iterations=iterations)
        # 2. Logistic Map Chaotic Permutation
        doubly_scrambled_matrix = logistic_map_scramble(scrambled_matrix, x, r)

        base_name = os.path.splitext(filename)[0]
        encrypted_filename = f'encrypted_{base_name}.png'
        encrypted_path = os.path.join(app.config['ENCRYPTED_FOLDER'], encrypted_filename)
        save_image_from_matrix(doubly_scrambled_matrix, encrypted_path)

        duration_ms = round((time.time() - start_time) * 1000, 1)
        metrics = compute_security_metrics(doubly_scrambled_matrix)
        ml_audit = predict_crypto_security(doubly_scrambled_matrix)

        return jsonify({
            'encrypted_image': encrypted_filename,
            'duration_ms': duration_ms,
            'dimensions': [doubly_scrambled_matrix.shape[1], doubly_scrambled_matrix.shape[0]],
            'metrics': metrics,
            'ml_audit': ml_audit
        }), 200

    except Exception as e:
        return jsonify({'error': f'Encryption error: {str(e)}'}), 500

@app.route('/decrypt', methods=['POST'])
def decrypt_image():
    start_time = time.time()
    data = request.json or {}
    encrypted_filename = data.get('encrypted_filename')
    original_filename = data.get('original_filename')
    x = data.get('x')
    r = data.get('r')
    iterations = data.get('iterations', 1)

    if not encrypted_filename or x is None or r is None:
        return jsonify({'error': 'Missing parameters for decryption'}), 400

    encrypted_path = os.path.join(app.config['ENCRYPTED_FOLDER'], encrypted_filename)
    if not os.path.exists(encrypted_path):
        # Allow decrypting a file uploaded directly to uploads folder
        encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_filename)

    if not os.path.exists(encrypted_path):
        return jsonify({'error': 'Encrypted image not found on server'}), 400

    try:
        x = float(x)
        r = float(r)
        iterations = max(1, min(20, int(iterations)))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid numeric parameters for x, r, or iterations'}), 400

    try:
        rgb_matrix = image_to_rgb_matrix(encrypted_path)
        if rgb_matrix.shape[0] != rgb_matrix.shape[1]:
            return jsonify({'error': 'Encrypted image must be square (NxN)'}), 400

        # 1. Inverse Logistic Chaotic Map Permutation
        descrambled_matrix = logistic_map_descramble(rgb_matrix, x, r)
        # 2. Inverse Arnold Cat Map
        decrypted_matrix = inverse_arnold_cat_map(descrambled_matrix, iterations=iterations)

        base_name = os.path.splitext(encrypted_filename)[0]
        decrypted_filename = f'decrypted_{base_name}.png'
        decrypted_path = os.path.join(app.config['DECRYPTED_FOLDER'], decrypted_filename)
        save_image_from_matrix(decrypted_matrix, decrypted_path)

        duration_ms = round((time.time() - start_time) * 1000, 1)
        metrics = compute_security_metrics(decrypted_matrix)
        ml_audit = predict_crypto_security(decrypted_matrix)

        # Integrity verification against original image if available
        integrity = {'verified': False, 'match': False, 'mse': None, 'psnr': None}
        if original_filename:
            orig_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            if os.path.exists(orig_path):
                orig_matrix = image_to_rgb_matrix(orig_path)
                if orig_matrix.shape == decrypted_matrix.shape:
                    is_match = bool(np.array_equal(orig_matrix, decrypted_matrix))
                    mse = float(np.mean((orig_matrix.astype(float) - decrypted_matrix.astype(float)) ** 2))
                    psnr = "Infinity" if mse == 0 else round(float(20 * np.log10(255.0 / np.sqrt(mse))), 2)
                    integrity = {
                        'verified': True,
                        'match': is_match,
                        'mse': round(mse, 4),
                        'psnr': psnr
                    }

        return jsonify({
            'decrypted_image': decrypted_filename,
            'duration_ms': duration_ms,
            'dimensions': [decrypted_matrix.shape[1], decrypted_matrix.shape[0]],
            'metrics': metrics,
            'ml_audit': ml_audit,
            'integrity': integrity
        }), 200

    except Exception as e:
        return jsonify({'error': f'Decryption error: {str(e)}'}), 500

@app.route('/batch_decrypt', methods=['POST'])
def batch_decrypt():
    """
    Decrypts an entire batch/folder of images and packages them into a single downloadable ZIP file.
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No images found in upload request'}), 400

    files = request.files.getlist('images')
    if not files or len(files) == 0 or (len(files) == 1 and files[0].filename == ''):
        return jsonify({'error': 'No valid image files were selected'}), 400

    x = request.form.get('x', '0.5888')
    r = request.form.get('r', '3.98888')
    iterations = request.form.get('iterations', '1')

    try:
        x = float(x)
        r = float(r)
        iterations = max(1, min(20, int(iterations)))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid key parameters (x, r, or iterations)'}), 400

    start_time = time.time()
    zip_buffer = io.BytesIO()
    decrypted_items = []
    processed_count = 0

    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            if not file or not file.filename:
                continue
            orig_name = secure_filename(os.path.basename(file.filename))
            if not orig_name:
                continue

            base_name, ext = os.path.splitext(orig_name)
            if ext.lower().lstrip('.') not in ALLOWED_EXTENSIONS:
                continue

            try:
                img = Image.open(file.stream).convert('RGB')
                w, h = img.size
                if w != h:
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    img = img.crop((left, top, left + min_dim, top + min_dim))

                rgb_mat = np.array(img)
                # 1. Inverse Logistic Chaotic Map
                descrambled = logistic_map_descramble(rgb_mat, x, r)
                # 2. Inverse Arnold Cat Map
                decrypted = inverse_arnold_cat_map(descrambled, iterations=iterations)

                # Save to PNG in memory
                out_img = Image.fromarray(np.uint8(decrypted))
                out_io = io.BytesIO()
                out_img.save(out_io, format='PNG')
                out_bytes = out_io.getvalue()

                out_filename = f"decrypted_{base_name}.png"
                zip_file.writestr(out_filename, out_bytes)

                # Also save to decrypted folder for preview
                saved_path = os.path.join(app.config['DECRYPTED_FOLDER'], out_filename)
                with open(saved_path, 'wb') as sf:
                    sf.write(out_bytes)

                decrypted_items.append({
                    'original': orig_name,
                    'decrypted': out_filename,
                    'preview_url': f"/decrypted/{out_filename}",
                    'size': f"{w}x{h}"
                })
                processed_count += 1
            except Exception as e:
                print(f"Skipping failed file {orig_name}: {e}")
                continue

    if processed_count == 0:
        return jsonify({'error': 'No supported images could be processed'}), 400

    zip_buffer.seek(0)
    bundle_name = f"decrypted_bundle_{int(time.time())}.zip"
    bundle_path = os.path.join(app.config['DECRYPTED_FOLDER'], bundle_name)
    with open(bundle_path, 'wb') as zf:
        zf.write(zip_buffer.getvalue())

    duration_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'success': True,
        'count': processed_count,
        'duration_ms': duration_ms,
        'zip_filename': bundle_name,
        'download_url': f"/decrypted/{bundle_name}",
        'items': decrypted_items
    }), 200

@app.route('/batch_encrypt', methods=['POST'])
def batch_encrypt():
    """
    Encrypts an entire batch/folder of images and packages them into a single downloadable ZIP file.
    """
    if 'images' not in request.files:
        return jsonify({'error': 'No images found in upload request'}), 400

    files = request.files.getlist('images')
    if not files or len(files) == 0 or (len(files) == 1 and files[0].filename == ''):
        return jsonify({'error': 'No valid image files were selected'}), 400

    x = request.form.get('x', '0.5888')
    r = request.form.get('r', '3.98888')
    iterations = request.form.get('iterations', '1')

    try:
        x = float(x)
        r = float(r)
        iterations = max(1, min(20, int(iterations)))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid key parameters (x, r, or iterations)'}), 400

    start_time = time.time()
    zip_buffer = io.BytesIO()
    encrypted_items = []
    processed_count = 0

    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            if not file or not file.filename:
                continue
            orig_name = secure_filename(os.path.basename(file.filename))
            if not orig_name:
                continue

            base_name, ext = os.path.splitext(orig_name)
            if ext.lower().lstrip('.') not in ALLOWED_EXTENSIONS:
                continue

            try:
                img = Image.open(file.stream).convert('RGB')
                w, h = img.size
                if w != h:
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    img = img.crop((left, top, left + min_dim, top + min_dim))

                rgb_mat = np.array(img)
                # 1. Arnold Cat Map Spatial Scrambling
                scrambled = arnold_cat_map(rgb_mat, iterations=iterations)
                # 2. Logistic Map Chaotic Permutation
                doubly_scrambled = logistic_map_scramble(scrambled, x, r)

                # Save to PNG in memory
                out_img = Image.fromarray(np.uint8(doubly_scrambled))
                out_io = io.BytesIO()
                out_img.save(out_io, format='PNG')
                out_bytes = out_io.getvalue()

                out_filename = f"encrypted_{base_name}.png"
                zip_file.writestr(out_filename, out_bytes)

                # Also save to encrypted folder for preview
                saved_path = os.path.join(app.config['ENCRYPTED_FOLDER'], out_filename)
                with open(saved_path, 'wb') as sf:
                    sf.write(out_bytes)

                encrypted_items.append({
                    'original': orig_name,
                    'processed': out_filename,
                    'preview_url': f"/encrypted/{out_filename}",
                    'size': f"{w}x{h}"
                })
                processed_count += 1
            except Exception as e:
                print(f"Skipping failed file {orig_name}: {e}")
                continue

    if processed_count == 0:
        return jsonify({'error': 'No supported images could be processed'}), 400

    zip_buffer.seek(0)
    bundle_name = f"encrypted_bundle_{int(time.time())}.zip"
    bundle_path = os.path.join(app.config['ENCRYPTED_FOLDER'], bundle_name)
    with open(bundle_path, 'wb') as zf:
        zf.write(zip_buffer.getvalue())

    duration_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'success': True,
        'count': processed_count,
        'duration_ms': duration_ms,
        'zip_filename': bundle_name,
        'download_url': f"/encrypted/{bundle_name}",
        'items': encrypted_items
    }), 200

@app.route('/ml_artifacts/<path:filename>')
def ml_artifacts_file(filename):
    return send_from_directory(ARTIFACTS_FOLDER, filename)

@app.route('/api/ml/benchmark', methods=['GET'])
def get_ml_benchmark():
    metrics_path = os.path.join(ARTIFACTS_FOLDER, 'metrics_summary.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Benchmark metrics not generated yet. Run train_crypto_classifier.py.'}), 404
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        data['plots'] = {
            'confusion_matrix': '/ml_artifacts/confusion_matrix.png',
            'feature_importance': '/ml_artifacts/feature_importance.png',
            'model_comparison': '/ml_artifacts/model_comparison.png'
        }
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/inspect', methods=['POST'])
def ml_inspect_image():
    """
    Accepts filename from uploads/encrypted/decrypted or a direct file upload
    and returns full ML security classification diagnostics.
    """
    mat = None
    if 'image' in request.files and request.files['image'].filename:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        mat = np.array(img)
    else:
        data = request.json or {}
        filename = data.get('filename')
        folder = data.get('folder', 'uploads')
        if filename:
            search_folders = [app.config['UPLOAD_FOLDER'], app.config['ENCRYPTED_FOLDER'], app.config['DECRYPTED_FOLDER']]
            for sf in search_folders:
                p = os.path.join(sf, filename)
                if os.path.exists(p):
                    mat = image_to_rgb_matrix(p)
                    break

    if mat is None:
        return jsonify({'error': 'Image not found for inspection'}), 400

    audit = predict_crypto_security(mat)
    if audit is None:
        return jsonify({'error': 'ML model not available'}), 503

    return jsonify({'success': True, 'audit': audit}), 200

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/encrypted/<path:filename>')
def encrypted_file(filename):
    is_zip = filename.lower().endswith('.zip')
    return send_from_directory(
        app.config['ENCRYPTED_FOLDER'],
        filename,
        as_attachment=is_zip,
        download_name=filename if is_zip else None
    )

@app.route('/decrypted/<path:filename>')
def decrypted_file(filename):
    is_zip = filename.lower().endswith('.zip')
    return send_from_directory(
        app.config['DECRYPTED_FOLDER'],
        filename,
        as_attachment=is_zip,
        download_name=filename if is_zip else None
    )

if __name__ == '__main__':
    print("Starting Classical Image Encryption Web App on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
