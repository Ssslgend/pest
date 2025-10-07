# sd_raster_prediction/train_raster.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys # Add sys import
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import csv
import datetime

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level from sd_raster_prediction
if project_root not in sys.path:
    sys.path.append(project_root)

# Assume these files are in the same directory or properly installed/pathed
from config_raster import get_config
from data_processor_raster import RasterPredictionDataProcessor
# from bilstm_adapted import BiLSTMClassifier # Make sure this is the adapted model
from model.bilstm import BiLSTMModel # Correct class name
from utils import evaluate_model, plot_training_history, save_checkpoint, load_checkpoint, calculate_and_plot_permutation_importance

class TrainingLogger:
    """记录训练过程中的指标，保存为CSV格式"""
    
    def __init__(self, log_dir, model_name="BiLSTM"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            model_name: 模型名称
        """
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{model_name.lower()}_training_log.csv")
        self.metrics_data = []
        
        # 初始化CSV表头
        self.headers = [
            "epoch", "train_loss", "train_auc", "val_loss", "val_auc", 
            "learning_rate", "time_elapsed", "Model"
        ]
        
        # 写入CSV头
        with open(self.log_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
        self.start_time = time.time()
        print(f"Training logger initialized. Logs will be saved to: {self.log_file}")
    
    def log_epoch(self, epoch, train_loss, train_auc, val_loss, val_auc, learning_rate):
        """记录每个epoch的训练指标"""
        time_elapsed = time.time() - self.start_time
        
        row_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "learning_rate": learning_rate,
            "time_elapsed": time_elapsed,
            "Model": self.model_name
        }
        
        self.metrics_data.append(row_data)
        
        # 写入CSV
        with open(self.log_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row_data)
            
        return row_data
    
    def save_to_comparison_dir(self, comparison_dir):
        """将训练日志复制到模型对比目录"""
        comparison_data_dir = os.path.join(comparison_dir, "data")
        os.makedirs(comparison_data_dir, exist_ok=True)
        
        target_file = os.path.join(comparison_data_dir, f"{self.model_name.lower()}_training_log.csv")
        
        # 将数据转换为DataFrame并保存
        df = pd.DataFrame(self.metrics_data)
        df.to_csv(target_file, index=False, encoding='utf-8-sig')
        print(f"Training logs copied to model comparison directory: {target_file}")

def train_epoch(model, loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    dataset_size = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels, _ in progress_bar:
        # inputs shape: (batch, 1, features)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # outputs shape: (batch, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().numpy().flatten())
        all_outputs.extend(outputs.detach().cpu().numpy().flatten())
        dataset_size += inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / dataset_size
    labels_np = np.array(all_labels)
    outputs_np = np.array(all_outputs)

    train_auc = 0.5
    if len(np.unique(labels_np)) >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(labels_np, outputs_np)
        except ValueError:
            print("Warning: Could not calculate train AUC.")
    else:
        print("Warning: Only one class present during training epoch, AUC is 0.5.")

    return avg_loss, train_auc

def main():
    print("--- Starting Raster Prediction Model Training --- ")
    start_time = time.time()

    # --- 1. Configuration and Setup --- ##配置加载
    CONFIG = get_config()
    DEVICE = CONFIG['training']['device']
    MODEL_SAVE_PATH = CONFIG['model_save_path']
    print(f"Using device: {DEVICE}")
    
    # 初始化日志记录器
    logs_dir = os.path.join(os.path.dirname(CONFIG['model_save_path']), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = TrainingLogger(logs_dir, "BiLSTM")

    # --- 2. Data Loading and Preparation --- ## 数据加载
    print("\n--- Loading and Preparing Data --- ")
    data_processor = RasterPredictionDataProcessor()
    try:
        data_splits = data_processor.load_prepare_and_split_data()
        input_size = data_splits['input_size']
        feature_names = data_splits['feature_names']
        print(f"Data loaded. Input size: {input_size}, Features: {feature_names}")
        train_loader, val_loader, test_loader = data_processor.get_dataloaders(CONFIG['training']['batch_size'])
    except Exception as e:
        print(f"Error during data processing: {e}")
        return # Stop execution if data loading fails

    # --- 3. Model Initialization --- ## 模型初始化
    print("\n--- Initializing Model --- ")
    # Prepare model config dictionary from the main config
    model_config = {
        "input_size": input_size,
        "hidden_size": CONFIG['model']['hidden_size'],
        "num_layers": CONFIG['model']['num_layers'],
        "dropout": CONFIG['model']['dropout'],
        # "num_classes" is handled by output_size argument below
    }
    # Instantiate using BiLSTMModel with config and output_size
    model = BiLSTMModel(
        config=model_config,
        output_size=CONFIG['model']['output_size'] # Pass desired output size explicitly
    ).to(DEVICE)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 4. Loss Function and Optimizer --- ## 损失和优化器
    # 使用BCEWithLogitsLoss替代BCELoss，更稳定且包含数值稳定性
    criterion = nn.BCEWithLogitsLoss() # 包含数值稳定性的二分类交叉熵损失

    # 使用AdamW优化器并调整参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # 监控验证集AUC，越高越好
        factor=0.5,          # 学习率调整因子
        patience=4,          # 在减小学习率前等待的epoch数（减少以更快响应）
        verbose=True,        # 打印学习率变化
        threshold=0.01,      # 改进的最小阈值
        min_lr=1e-6          # 最小学习率
    )

    # --- 5. Training Loop --- ## 训练循环
    print("\n--- Starting Training --- ")
    num_epochs = CONFIG['training']['num_epochs']
    patience = CONFIG['training']['patience']
    best_val_auc = -np.inf
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    
    # 调整过拟合检测的变量
    overfit_threshold = 0.15  # 提高差异阈值，从0.12提高到0.15
    consecutive_overfit = 0   # 连续过拟合轮次计数
    max_consecutive_overfit = 3  # 增加允许的最大连续过拟合轮次，从2增加到3
    
    # 调整AUC过高检测
    max_val_auc = 0.92  # 提高验证AUC的最大允许值，从0.95降到0.92
    auc_too_high_count = 0   # 验证AUC过高计数
    max_auc_too_high = 4     # 增加允许连续验证AUC过高的最大次数，从3增加到4

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)

        # Validate
        val_metrics, _, _, _ = evaluate_model(model, val_loader, DEVICE, criterion)
        val_loss = val_metrics['loss']
        val_auc = val_metrics['auc']
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练指标到日志文件
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_auc=train_auc,
            val_loss=val_loss,
            val_auc=val_auc,
            learning_rate=current_lr
        )

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} Summary | Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val AUC:   {val_auc:.4f}")

        # 检测过拟合 - 计算训练和验证AUC的差异
        auc_diff = train_auc - val_auc
        print(f"  Train-Val AUC差异: {auc_diff:.4f}")
        
        # 检测AUC是否过高（过拟合的另一个指标）
        if val_auc > max_val_auc:
            auc_too_high_count += 1
            print(f"  警告: 验证AUC过高 ({val_auc:.4f} > {max_val_auc:.4f})，可能过拟合! (连续{auc_too_high_count}次)")
        else:
            auc_too_high_count = 0
            
        if auc_diff > overfit_threshold:
            consecutive_overfit += 1
            print(f"  警告: 可能过拟合! 训练AUC比验证AUC高 {auc_diff:.4f} (连续{consecutive_overfit}次)")
        else:
            consecutive_overfit = 0
        
        # 过拟合提前停止条件
        if consecutive_overfit >= max_consecutive_overfit:
            print(f"  连续{consecutive_overfit}次出现过拟合，训练提前停止")
            break
            
        # AUC过高提前停止条件  
        if auc_too_high_count >= max_auc_too_high:
            print(f"  验证AUC连续{auc_too_high_count}次超过{max_val_auc:.4f}，可能存在严重过拟合，训练提前停止")
            break

        # 更新学习率调度器
        scheduler.step(val_auc)  # 基于验证集AUC调整学习率
        print(f"  Current Learning Rate: {current_lr:.6f}")

        # 改进的检查点保存和早停策略
        # 修改评分方式，平衡惩罚过拟合和重视验证集性能
        auc_penalty = 0
        if val_auc > max_val_auc:
            # 减轻对AUC过高的惩罚，从0.5降到0.3
            auc_penalty = (val_auc - max_val_auc) * 0.3
            
        # 调整评分权重，更重视验证集AUC
        val_score = val_auc * 0.7 + (1.0 - val_loss) * 0.15 - auc_diff * 0.15 - auc_penalty

        if val_score > best_val_auc:
            print(f"验证集性能改进 ({best_val_auc:.4f} --> {val_score:.4f}). 保存模型...")
            best_val_auc = val_score
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 保存调度器状态
                'best_val_score': best_val_auc,
                'input_size': input_size, # 保存输入尺寸以便预测
                'feature_names': feature_names # 保存特征名称
            }, MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"验证集性能已连续 {epochs_no_improve} 个epoch未改善. 当前最佳: {best_val_auc:.4f}")

            # 动态调整早停耐心值
            # 如果学习率已经很低，减少耐心值
            effective_patience = patience
            if current_lr <= 1e-5:
                effective_patience = max(3, patience // 2)  # 当学习率很低时减少耐心值
                print(f"  学习率很低，调整早停耐心值为: {effective_patience}")
            
            # 如果有过拟合迹象，减少耐心值但不要太激进
            if auc_diff > overfit_threshold * 0.9 or val_auc > max_val_auc * 0.97:  # 接近过拟合阈值
                effective_patience = max(2, effective_patience // 2)
                print(f"  检测到过拟合迹象，调整早停耐心值为: {effective_patience}")

            if epochs_no_improve >= effective_patience:
                print(f"早停触发，在第 {epoch + 1} 个epoch后停止训练.")
                break

    # --- 6. Final Evaluation on Test Set --- ## 测试集评估
    print("\n--- Training Finished. Evaluating on Test Set --- ")
    # Load the best model
    checkpoint, _ = load_checkpoint(MODEL_SAVE_PATH, model, device=DEVICE)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model (Epoch {checkpoint.get('epoch', 'N/A')}, Val AUC: {checkpoint.get('best_val_auc', -1):.4f}) loaded.")
    else:
        print("Warning: Could not load best model checkpoint for final evaluation.")

    test_metrics, test_labels, test_outputs, test_coords = evaluate_model(model, test_loader, DEVICE, criterion)

    print("\n--- Test Set Evaluation Results --- ")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test AUC:  {test_metrics['auc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # --- 7. Plotting and Feature Importance --- ## 结果分析
    print("\n--- Generating Plots and Calculating Feature Importance --- ")
    # Plot training history
    plot_training_history(history, CONFIG['training_history_plot_path'])
    
    # 将日志文件复制到model_comparison_v2/data目录
    try:
        comparison_dir = os.path.join(project_root, "model_comparison_v2")
        if os.path.exists(comparison_dir):
            logger.save_to_comparison_dir(comparison_dir)
    except Exception as e:
        print(f"Warning: Could not copy logs to comparison directory: {e}")

    # Calculate Feature Importance
    try:
        # Pass data_processor which holds the test data and feature names
        calculate_and_plot_permutation_importance(model, data_processor, test_metrics['auc'], DEVICE)
    except Exception as e:
        print(f"Error calculating/plotting feature importance: {e}")

    total_duration = time.time() - start_time
    print(f"\n--- Training and Evaluation Complete --- ")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Training logs saved to: {logger.log_file}")
    print(f"Scaler saved to: {CONFIG['scaler_save_path']}")
    print(f"Results and plots saved in: {CONFIG['prediction_output_dir']} (and subdirectories)")

if __name__ == '__main__':
    # Ensure the adapted bilstm model file exists and is importable
    # if not os.path.exists('sd_raster_prediction/bilstm_adapted.py'):
    #     print("Error: `sd_raster_prediction/bilstm_adapted.py` not found. Please copy and rename the model file.")
    # else:
    main()