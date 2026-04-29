import numpy as np
import joblib
from PIL import Image
import base64
import io
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载sklearn模型
model = None
if os.path.exists('model.pth'):
    model = joblib.load('model.pth')
    print("Model loaded from model.pth")
else:
    print("Warning: model.pth not found")

def center_image(img_array):
    """将数字居中对齐"""
    rows = np.any(img_array > 0.1, axis=1)
    cols = np.any(img_array > 0.1, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img_array
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    digit = img_array[y_min:y_max+1, x_min:x_max+1]
    height, width = digit.shape
    max_dim = max(height, width)
    scale = 20 / max_dim
    
    new_height, new_width = int(height * scale), int(width * scale)
    from PIL import Image as PILImage
    digit_img = PILImage.fromarray((digit * 255).astype('uint8'))
    digit_img = digit_img.resize((new_width, new_height), PILImage.LANCZOS)
    
    centered = np.zeros((28, 28))
    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2
    
    centered[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = np.array(digit_img) / 255.0
    
    return centered

def predict_digit(image):
    """预测手写数字"""
    try:
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        img_array = np.where(img_array < 128, 0, 255)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = center_image(img_array)
        
        img_flat = img_array.flatten().reshape(1, -1)
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]
        
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_results = [(int(i), float(probabilities[i])) for i in top3_indices]
        all_probabilities = [float(probabilities[i]) for i in range(10)]
        
        return {
            'prediction': int(prediction),
            'confidence': float(probabilities[prediction]),
            'top3': top3_results,
            'all_probabilities': all_probabilities
        }
    except Exception as e:
        return {'error': str(e)}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>手写数字识别</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; color: white; margin-bottom: 30px; font-size: 28px; }
        .main-content { display: grid; grid-template-columns: 1fr 1.2fr; gap: 25px; }
        .card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 25px;
        }
        .card-title { color: #333; margin-bottom: 18px; font-size: 18px; }
        .canvas-container {
            border: 3px solid #ddd;
            border-radius: 15px;
            overflow: hidden;
        }
        #canvas { display: block; background: white; cursor: crosshair; }
        .canvas-buttons { display: flex; gap: 12px; margin-top: 18px; }
        .btn {
            flex: 1;
            padding: 12px 18px;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); }
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        .btn-secondary:hover { background: #e0e0e0; }
        .result-section { min-height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .result-placeholder { color: #999; text-align: center; }
        .prediction-box { text-align: center; margin-bottom: 20px; }
        .prediction-number {
            font-size: 64px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        .prediction-confidence { font-size: 16px; color: #666; }
        .probability-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            margin-top: 20px;
        }
        .prob-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            border: 2px solid transparent;
        }
        .prob-item.highlight {
            background: #eef2ff;
            border-color: #667eea;
        }
        .prob-digit { font-size: 20px; font-weight: bold; color: #333; }
        .prob-value { font-size: 11px; color: #666; margin-top: 5px; }
        .top3-section { margin-top: 20px; padding-top: 15px; border-top: 1px solid #e0e0e0; }
        .top3-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .top3-rank {
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 12px;
            color: white;
        }
        .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); }
        .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); }
        .rank-3 { background: linear-gradient(135deg, #CD7F32, #B87333); }
        .top3-digit { font-size: 20px; font-weight: bold; color: #333; margin-right: auto; }
        .top3-confidence { font-size: 13px; color: #666; }
        .processing { display: flex; flex-direction: column; align-items: center; gap: 12px; }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .hint {
            background: #fff3cd;
            color: #856404;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            margin-top: 15px;
        }
        @media (max-width: 768px) {
            .main-content { grid-template-columns: 1fr; }
            h1 { font-size: 22px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✏️ 手写数字识别</h1>
        <div class="main-content">
            <div class="card">
                <h2 class="card-title">🎨 手写区域</h2>
                <div class="canvas-container">
                    <canvas id="canvas" width="280" height="280"></canvas>
                </div>
                <div class="canvas-buttons">
                    <button class="btn btn-secondary" onclick="clearCanvas()">🗑️ 清空</button>
                    <button class="btn btn-primary" onclick="recognizeDigit()">🔍 识别</button>
                </div>
                <div class="hint">
                    <strong>💡 提示：</strong>用鼠标或触屏手写数字，尽量写大一些，占据画布中心区域。
                </div>
            </div>
            <div class="card">
                <h2 class="card-title">📊 识别结果</h2>
                <div class="result-section" id="resultSection">
                    <div class="result-placeholder">
                        <div style="font-size: 40px; margin-bottom: 10px;">✏️</div>
                        <p>请在左侧画布上手写数字，然后点击「识别」按钮</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        let isDrawing = false;
        let lastX = 0, lastY = 0;

        canvas.addEventListener('mousedown', startDraw);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDraw);
        canvas.addEventListener('mouseout', stopDraw);
        canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); });
        canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
        canvas.addEventListener('touchend', stopDraw);

        function startDraw(e) {
            isDrawing = true;
            const pos = getPos(e);
            [lastX, lastY] = pos;
        }

        function draw(e) {
            if (!isDrawing) return;
            const [x, y] = getPos(e);
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        }

        function stopDraw() { isDrawing = false; }

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return [clientX - rect.left, clientY - rect.top];
        }

        function clearCanvas() {
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('resultSection').innerHTML = `
                <div class="result-placeholder">
                    <div style="font-size: 40px; margin-bottom: 10px;">✏️</div>
                    <p>请在左侧画布上手写数字，然后点击「识别」按钮</p>
                </div>
            `;
        }

        async function recognizeDigit() {
            document.getElementById('resultSection').innerHTML = `
                <div class="processing">
                    <div class="spinner"></div>
                    <p>正在识别...</p>
                </div>
            `;

            try {
                const imageData = canvas.toDataURL('image/png');
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                const result = await response.json();

                if (result.error) throw new Error(result.error);

                const probGrid = result.all_probabilities.map((prob, i) => {
                    const isHighlight = i === result.prediction;
                    return `<div class="prob-item ${isHighlight ? 'highlight' : ''}">
                        <div class="prob-digit">${i}</div>
                        <div class="prob-value">${(prob * 100).toFixed(0)}%</div>
                    </div>`;
                }).join('');

                const top3 = result.top3.map((item, i) => `
                    <div class="top3-item">
                        <div class="top3-rank rank-${i + 1}">${i + 1}</div>
                        <div class="top3-digit">${item[0]}</div>
                        <div class="top3-confidence">${(item[1] * 100).toFixed(1)}%</div>
                    </div>
                `).join('');

                document.getElementById('resultSection').innerHTML = `
                    <div class="prediction-box">
                        <div class="prediction-number">${result.prediction}</div>
                        <div class="prediction-confidence">📈 置信度: ${(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div style="width: 100%;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">📊 0-9 概率分布</div>
                        <div class="probability-grid">${probGrid}</div>
                    </div>
                    <div class="top3-section">
                        <div style="font-size: 14px; color: #666; margin-bottom: 10px;">🏆 Top-3 预测</div>
                        ${top3}
                    </div>
                `;
            } catch (error) {
                document.getElementById('resultSection').innerHTML = `
                    <div style="color: #dc2626;">
                        <div style="font-size: 40px; margin-bottom: 10px;">❌</div>
                        <p>识别失败: ${error.message}</p>
                    </div>
                `;
            }
        }

        clearCanvas();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')
        
        result = predict_digit(image)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)