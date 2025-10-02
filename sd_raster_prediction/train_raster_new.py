# sd_raster_prediction/train_raster_new.py
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
import torch.nn.functional as F

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level from sd_raster_prediction
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用新的配置文件和数据处理器
from sd_raster_prediction.config_raster_new import get_config
from .data_processor_raster_new import RasterPredictionDataProcessor
from model.bilstm import BiLSTMModel
from .utils import (
    evaluate_model, plot_training_history, save_checkpoint, load_checkpoint, 
    calculate_and_plot_permutation_importance,
    # 导入新增的可视化函数
    plot_confusion_matrix, plot_roc_curve, plot_metrics_comparison,
    plot_threshold_impact, plot_top_features_impact, plot_feature_interactions
)

# 导入必要的可视化函数
from sd_raster_prediction.utils import plot_confusion_matrix, plot_roc_curve, plot_training_history

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
    """Training function for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    dataset_size = 0
    
    # 跟踪预测概率分布 - 确保与DistributionRegularizationLoss中的bins参数一致
    num_bins = 20  # 必须与DistributionRegularizationLoss中的bins参数相同
    bin_edges = np.linspace(0, 1, num_bins+1)  # 0-1之间划分num_bins个区间
    bin_counts = np.zeros(num_bins)

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels, _ in progress_bar:
        # inputs shape: (batch, 1, features)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # outputs shape: (batch, 1)
        
        # 如果使用自定义损失函数，获取详细统计信息
        if hasattr(criterion, 'base_criterion'):
            loss, loss_stats = criterion(outputs, labels, return_stats=True)
            # 更新区间统计
            bin_counts += loss_stats['bin_counts']
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.cpu().numpy().flatten())
        
        # 记录sigmoid后的概率
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_outputs.extend(probs)
        dataset_size += inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / dataset_size
    labels_np = np.array(all_labels)
    outputs_np = np.array(all_outputs)
    
    # 打印概率分布信息
    print("\n预测概率分布情况:")
    for i in range(num_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_count = np.sum((outputs_np >= bin_start) & (outputs_np < bin_end))
        bin_percent = (bin_count / len(outputs_np)) * 100
        print(f"  概率区间 [{bin_start:.1f}-{bin_end:.1f}): {bin_count} ({bin_percent:.2f}%)")

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

# 添加自定义均匀分布损失函数，促使概率分布更加均匀

class DistributionRegularizationLoss(nn.Module):
    """
    自定义损失函数，鼓励模型产生更加均匀的预测概率分布
    """
    def __init__(self, base_criterion, bins=10, lambda_reg=0.3):
        super(DistributionRegularizationLoss, self).__init__()
        self.base_criterion = base_criterion  # 基础分类损失函数
        self.bins = bins  # 概率区间数量
        self.lambda_reg = lambda_reg  # 均匀化正则项权重
        
    def forward(self, predictions, targets, return_stats=False):
        # 基础分类损失
        base_loss = self.base_criterion(predictions, targets)
        
        # 计算预测概率的分布情况
        probs = torch.sigmoid(predictions).detach()
        
        # 创建概率区间
        bin_edges = torch.linspace(0, 1, self.bins + 1, device=probs.device)
        bin_counts = torch.zeros(self.bins, device=probs.device)
        
        # 统计每个区间的样本数量
        for i in range(self.bins):
            if i == self.bins - 1:
                bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
            else:
                bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            bin_counts[i] = bin_mask.sum().float()
        
        # 计算分布不均匀性，添加数值稳定性处理
        if bin_counts.sum() > 0:
            bin_probs = bin_counts / bin_counts.sum()
            
            # 添加小的平滑值以避免log(0)
            epsilon = 1e-7
            bin_probs = bin_probs + epsilon
            bin_probs = bin_probs / bin_probs.sum()  # 重新归一化
            
            # 理想的均匀分布
            uniform_probs = torch.ones_like(bin_probs) / self.bins
            
            # 使用更稳定的方法计算KL散度
            # 使用torch.clamp防止log接收到0值
            kl_div = torch.sum(bin_probs * torch.log(torch.clamp(bin_probs / uniform_probs, min=epsilon)))
            
            # 限制KL散度值，防止过大
            kl_div = torch.clamp(kl_div, max=10.0)
            
            # 结合基础损失和分布正则项，并确保结果为有限值
            reg_loss = self.lambda_reg * kl_div
            # 检查基础损失是否为无穷大，如果是则只使用正则项
            if torch.isinf(base_loss) or torch.isnan(base_loss):
                combined_loss = reg_loss
            else:
                combined_loss = base_loss + reg_loss
            
            # 最后再次检查确保损失是有限值
            if torch.isinf(combined_loss) or torch.isnan(combined_loss):
                combined_loss = torch.tensor(1.0, device=combined_loss.device, requires_grad=True)
        else:
            combined_loss = base_loss
            kl_div = torch.tensor(0.0, device=base_loss.device)
        
        if return_stats:
            stats = {
                'base_loss': base_loss.item() if not torch.isinf(base_loss) and not torch.isnan(base_loss) else 0.0,
                'bin_counts': bin_counts.cpu().numpy(),
                'kl_div': kl_div.item() if 'kl_div' in locals() and not torch.isinf(kl_div) and not torch.isnan(kl_div) else 0.0
            }
            return combined_loss, stats
        
        return combined_loss
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    print("--- Starting Raster Prediction Model Training --- ")
    start_time = time.time()

    # --- 1. Configuration and Setup --- ##配置加载
    CONFIG = get_config() #返回配置参数
    DEVICE = CONFIG['training']['device']
    MODEL_SAVE_PATH = CONFIG['model_save_path']
    print(f"Using device: {DEVICE}")
    
    # 初始化日志记录器
    logs_dir = os.path.join(os.path.dirname(CONFIG['model_save_path']), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = TrainingLogger(logs_dir, "SF-BiLSTM")

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
    # 使用基础BCEWithLogitsLoss作为分类损失
    base_criterion = nn.BCEWithLogitsLoss()
    
    # 使用自定义均匀分布损失函数包装基础损失
    criterion = DistributionRegularizationLoss(
        base_criterion=base_criterion, 
        bins=20,               # 将概率空间分为20个区间
        lambda_reg=0.5         # 分布正则化权重
    )
    
    print("使用概率分布均匀化正则损失函数")

    # 使用AdamW优化器并调整参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'] * 0.5,  # 减少权重衰减
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
    overfit_threshold = 0.20  # 增加差异阈值，让模型有更多训练机会
    consecutive_overfit = 0   # 连续过拟合轮次计数
    max_consecutive_overfit = 5  # 增加允许的最大连续过拟合轮次
    
    # 调整AUC过高检测
    max_val_auc = 0.98  # 增加验证AUC的最大允许值，因为不再生成伪阴性点，真实数据可能达到较高准确率
    auc_too_high_count = 0   # 验证AUC过高计数
    max_auc_too_high = 6     # 增加允许连续验证AUC过高的最大次数

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
        
        # 确保记录所有值，即使是inf或nan
        if val_loss is None:
            val_loss = float('nan')  # 使用nan代替None，保证历史记录连续性
            
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练指标到日志文件
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss if not np.isinf(train_loss) and not np.isnan(train_loss) else 0.0,
            train_auc=train_auc,
            val_loss=val_loss if not np.isinf(val_loss) and not np.isnan(val_loss) else 0.0,
            val_auc=val_auc,
            learning_rate=current_lr
        )

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_time:.2f}s")

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            
            # Save model checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_auc': val_auc,
                'optimizer': optimizer.state_dict(),
            }, MODEL_SAVE_PATH)
            print(f"Checkpoint saved. Best validation AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation AUC for {epochs_no_improve} epochs.")

        # 检查过拟合: 训练AUC明显高于验证AUC
        if train_auc - val_auc > overfit_threshold:
            consecutive_overfit += 1
            print(f"Warning: Potential overfitting detected. Train AUC exceeds Val AUC by {train_auc - val_auc:.4f}.")
            if consecutive_overfit >= max_consecutive_overfit:
                print(f"Stopping training due to consistent overfitting for {consecutive_overfit} epochs.")
                break
        else:
            consecutive_overfit = 0  # 重置

        # 检查验证AUC是否过高（可能表明数据泄露或样本不平衡问题）
        if val_auc > max_val_auc:
            auc_too_high_count += 1
            print(f"Warning: Validation AUC is suspiciously high ({val_auc:.4f}), check for data leakage.")
            if auc_too_high_count >= max_auc_too_high:
                print(f"Stopping training due to consistently high validation AUC for {auc_too_high_count} epochs.")
                break
        else:
            auc_too_high_count = 0  # 重置

        # 更新学习率调度器
        scheduler.step(val_auc)  # 使用验证AUC来调整学习率

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # --- 6. Final Model Evaluation --- ## 模型评估
    print("\n--- Final Model Evaluation --- ")
    # Load best model for final evaluation
    best_model_checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    model.eval()

    # Evaluate on test set
    test_metrics, test_preds, test_labels, test_coords = evaluate_model(model, test_loader, DEVICE, criterion)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Test AUC: {test_metrics['auc']:.4f}, "
          f"Test Accuracy: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}")

    # --- 7. Feature Importance Analysis --- ## 特征重要性分析
    print("\n--- Calculating Feature Importance --- ")
    try:
        # 创建一个简化版的数据处理器用于特征重要性计算
        from sd_raster_prediction.data_processor_raster_new import SdPestPresenceAbsenceDataset
        
        # 获取测试数据和特征名称
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        coords_test = data_splits['coords_test']
        
        # 获取测试集的基准AUC性能作为对比基准
        baseline_auc = test_metrics['auc']
        
        # 创建一个简单的数据处理器类用于特征重要性计算
        class SimpleDataProcessor:
            def __init__(self, X, y, coords, feature_names):
                self.data_dict = {'test': {'X': X, 'y': y, 'coordinates': coords}}
                self.feature_columns = feature_names
        
        # 创建简单数据处理器
        simple_processor = SimpleDataProcessor(X_test, y_test, coords_test, feature_names)
        
        # 使用正确的参数调用特征重要性函数
        importance_results = calculate_and_plot_permutation_importance(
            model, 
            simple_processor,  # 传递简化版数据处理器
            baseline_auc,      # 传递基准AUC
            DEVICE,            # 设备
            plot_path=CONFIG['feature_importance_plot_path'],
            csv_path=CONFIG['feature_importance_csv_path']
        )
        print(f"Feature importance analysis completed. Results saved to {CONFIG['feature_importance_plot_path']} and {CONFIG['feature_importance_csv_path']}")
    except Exception as e:
        print(f"Error during feature importance calculation: {e}")
        import traceback
        traceback.print_exc()

    # --- 8. Training Visualization --- ## 训练可视化
    print("\n--- Generating Training History Plot --- ")
    # 使用绘图函数
    # 这里我们将只使用标准的绘图函数
    loss_path = CONFIG['training_history_plot_path'].replace('.png', '_loss.png')
    auc_path = CONFIG['training_history_plot_path'].replace('.png', '_auc.png')
    plot_training_history(
        history, 
        save_path=CONFIG['training_history_plot_path']
    )
    print(f"Training history plots saved to {loss_path} and {auc_path}")
    print(f"Combined training history saved to {CONFIG['training_history_plot_path']}")

    # --- 新增: 9. 补充模型评估可视化 --- ##
    print("\n--- Generating Additional Model Evaluation Visualizations ---")
    try:
        # 创建可视化目录
        viz_dir = os.path.join(os.path.dirname(CONFIG['model_save_path']), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 确保数据类型正确
        test_labels_np = np.array(test_labels).astype(np.int32)  # 转换为整数
        # 使用sigmoid函数转换预测值为概率
        from scipy.special import expit as sigmoid
        test_probs = sigmoid(test_preds)  # 转换为概率
        y_pred_binary = (test_probs >= 0.5).astype(np.int32)  # 转换为二进制标签
        
        # 1. 混淆矩阵可视化
        confusion_matrix_path = os.path.join(viz_dir, 'confusion_matrix.png')
        plot_confusion_matrix(test_labels_np, y_pred_binary, confusion_matrix_path)
        
        # 2. ROC曲线可视化
        roc_curve_path = os.path.join(viz_dir, 'roc_curve.png')
        try:
            # 确保标签为二分类格式
            unique_labels = np.unique(test_labels_np)
            if len(unique_labels) > 2:
                print(f"警告: 发现多于两个类别: {unique_labels}")
                # 转换为二分类问题
                test_labels_binary = (test_labels_np == np.max(unique_labels)).astype(np.int32)
            else:
                test_labels_binary = test_labels_np
            
            # 尝试绘制ROC曲线
            plot_roc_curve(test_labels_binary, test_probs, roc_curve_path)
        except Exception as e:
            print(f"ROC曲线绘制错误: {e}")
            # 创建错误提示图像
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error plotting ROC curve:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            plt.axis('off')
            plt.savefig(roc_curve_path)
            plt.close()
        
        # 3. 模型评估指标比较
        metrics_path = os.path.join(viz_dir, 'model_metrics.png')
        plot_metrics_comparison(test_metrics, metrics_path)
        
        # 4. 阈值对各指标的影响
        threshold_path = os.path.join(viz_dir, 'threshold_impact.png')
        try:
            # 使用前面创建的二分类标签
            plot_threshold_impact(test_labels_binary, test_probs, threshold_path)
        except Exception as e:
            print(f"阈值分析图绘制错误: {e}")
            # 创建错误提示图像
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error plotting threshold impact:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            plt.axis('off')
            plt.savefig(threshold_path)
            plt.close()
        
        # 5. 前五个重要特征的影响分析
        if importance_results:
            feature_impact_path = os.path.join(viz_dir, 'top_features_impact.png')
            X_test = data_splits['X_test']
            y_test = data_splits['y_test']
            plot_top_features_impact(model, X_test, y_test, feature_names, 
                                    importance_results, feature_impact_path)
        
            # 6. 特征交互分析
            feature_interactions_path = os.path.join(viz_dir, 'feature_interactions.png')
            plot_feature_interactions(X_test, feature_names, importance_results, 
                                     feature_interactions_path)
        
        print(f"All additional visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Error during additional visualizations: {e}")
        import traceback
        traceback.print_exc()

    # --- 10. Finish --- ## 完成
    total_time = time.time() - start_time
    print(f"\n--- Training Complete ---")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {CONFIG['scaler_save_path']}")

if __name__ == '__main__':
    main() 