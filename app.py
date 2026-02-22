import sys
import os
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# 确保能导入 snake 文件夹下的定义
sys.path.append(os.path.join(os.path.dirname(__file__), 'snake'))
from snake_v6_long_train import HybridDuelingQNet

app = Flask(__name__)
CORS(app)

# 配置参数（需与训练脚本一致）
MODEL_PATH = 'snake/snake_v6_rust_best.pth'
device = torch.device("cpu")

# --- 初始化并加载模型 ---
def load_trained_model():
    model = HybridDuelingQNet()
    if os.path.exists(MODEL_PATH):
        # 核心修复：加载 checkpoint 字典中的 'model' 键
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"✅ 成功从 Checkpoint 加载模型权重 (步数: {checkpoint.get('steps', '未知')})")
    else:
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")

    model.to(device)
    model.eval()
    return model

model = load_trained_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # V6 需要两个输入：spatial (20x20x3) 和 aux (21)
        spatial_np = np.array(data['spatial'], dtype=np.float32)
        aux_np = np.array(data['aux'], dtype=np.float32)

        # 转换为 Tensor 并增加 Batch 维度
        spatial_t = torch.from_numpy(spatial_np).unsqueeze(0).to(device)
        aux_t = torch.from_numpy(aux_np).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(spatial_t, aux_t)
            action = q_values.argmax(dim=1).item()

        return jsonify({
            'action': action,
            'q_values': q_values.tolist()[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)