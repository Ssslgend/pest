import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

class ModelLogger:
    """
    用于记录模型训练过程中的各项指标，并输出标准格式的训练日志文件
    """

    def __init__(self, model_name, output_dir, config=None):
        """
        初始化训练日志记录器

        Args:
            model_name: 模型名称 ('MLP', 'LSTM', 'BiLSTM' 等)
            output_dir: 输出目录路径
            config: 模型配置信息（可选）
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rate': [],
            'time_elapsed': []
        }
        self.start_time = time.time()
        self.epoch_start_time = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置日志文件路径
        self.log_file_path = os.path.join(output_dir, f"{model_name.lower()}.log")
        self.csv_file_path = os.path.join(output_dir, "training_log.csv")

        # 配置日志记录器
        self.logger = logging.getLogger(f"{model_name}Logger")
        self.logger.setLevel(logging.INFO)

        # 清除现有处理器以避免重复记录
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s') # 简化控制台输出
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 记录初始化信息
        self.logger.info(f"=== {model_name} 训练日志 ===")
        self.logger.info(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if config:
            self.logger.info(f"模型配置: {config}")
        self.logger.info("---")

    def start_epoch(self, epoch):
        """记录训练轮次开始"""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        self.logger.info(f"\nEpoch {epoch} 开始")

    def end_epoch(self, epoch, train_loss, train_auc, val_loss, val_auc, learning_rate=None):
        """记录训练轮次结束及各项指标"""
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_duration = time.time() - self.start_time

        # 确保指标不是None或NaN，如果是则用np.nan代替
        train_loss = train_loss if pd.notna(train_loss) else np.nan
        train_auc = train_auc if pd.notna(train_auc) else np.nan
        val_loss = val_loss if pd.notna(val_loss) else np.nan
        val_auc = val_auc if pd.notna(val_auc) else np.nan
        learning_rate = learning_rate if pd.notna(learning_rate) else np.nan

        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_auc'].append(train_auc)
        self.metrics['val_auc'].append(val_auc)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['time_elapsed'].append(total_duration)

        # 记录到日志
        log_msg = (
            f"Epoch {epoch} 完成 | 耗时: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )
        if pd.notna(learning_rate):
            log_msg += f" | LR: {learning_rate:.6f}"
        self.logger.info(log_msg)

        # 保存CSV
        self._save_csv()

    def log_message(self, message, level='info'):
        """记录自定义消息"""
        if level.lower() == 'info':
            self.logger.info(message)
        elif level.lower() == 'warning':
            self.logger.warning(message)
        elif level.lower() == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

    def log_metrics(self, metrics_dict, prefix=''):
        """记录额外的指标信息"""
        message = prefix + " " if prefix else ""
        message += " | ".join([f"{k}: {v:.4f}" if isinstance(v, (float, np.number)) else f"{k}: {v}"
                               for k, v in metrics_dict.items()])
        self.logger.info(message)

    def _save_csv(self):
        """保存训练指标为CSV文件"""
        try:
            df = pd.DataFrame(self.metrics)
            df['Model'] = self.model_name # 添加模型标识列
            df.to_csv(self.csv_file_path, index=False, encoding='utf-8-sig') # 使用utf-8-sig确保Excel兼容
        except Exception as e:
            self.logger.error(f"保存CSV文件失败: {e}")

    def plot_training_history(self, save_path=None):
        """绘制训练历史曲线"""
        if not self.metrics['epoch']:
            self.logger.warning("没有训练历史数据可供绘图")
            return

        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{self.model_name.lower()}_training_history.png")

        # 配置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
             self.logger.warning(f"设置中文字体失败: {e}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 绘制损失曲线
        axes[0].plot(self.metrics['epoch'], self.metrics['train_loss'], 'b-o', markersize=4, label='训练损失')
        axes[0].plot(self.metrics['epoch'], self.metrics['val_loss'], 'r-s', markersize=4, label='验证损失')
        axes[0].set_title(f'{self.model_name} 训练和验证损失')
        axes[0].set_xlabel('轮次')
        axes[0].set_ylabel('损失')
        axes[0].legend()
        axes[0].grid(True)

        # 绘制AUC曲线
        axes[1].plot(self.metrics['epoch'], self.metrics['train_auc'], 'b-o', markersize=4, label='训练 AUC')
        axes[1].plot(self.metrics['epoch'], self.metrics['val_auc'], 'r-s', markersize=4, label='验证 AUC')
        axes[1].set_title(f'{self.model_name} 训练和验证 AUC')
        axes[1].set_xlabel('轮次')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        try:
            plt.savefig(save_path, dpi=300)
            self.logger.info(f"训练历史图表已保存至: {save_path}")
        except Exception as e:
            self.logger.error(f"保存训练历史图表失败: {e}")
        plt.close()

    def finish_training(self):
        """完成训练，记录总结信息并绘制图表"""
        total_time = time.time() - self.start_time
        total_epochs = len(self.metrics['epoch'])

        if not total_epochs:
            self.logger.warning("未记录任何训练数据")
            return

        # 找出最佳验证AUC
        valid_val_auc = [auc for auc in self.metrics['val_auc'] if pd.notna(auc)]
        if valid_val_auc:
            best_val_auc_idx = np.argmax(valid_val_auc)
            best_val_auc = valid_val_auc[best_val_auc_idx]
            # Find the corresponding epoch for the best valid AUC
            best_val_auc_epoch = [self.metrics['epoch'][i] for i, auc in enumerate(self.metrics['val_auc']) if pd.notna(auc)][best_val_auc_idx]
        else:
            best_val_auc = np.nan
            best_val_auc_epoch = np.nan

        # 找出最低验证损失
        valid_val_loss = [loss for loss in self.metrics['val_loss'] if pd.notna(loss)]
        if valid_val_loss:
            best_val_loss_idx = np.argmin(valid_val_loss)
            best_val_loss = valid_val_loss[best_val_loss_idx]
            best_val_loss_epoch = [self.metrics['epoch'][i] for i, loss in enumerate(self.metrics['val_loss']) if pd.notna(loss)][best_val_loss_idx]
        else:
            best_val_loss = np.nan
            best_val_loss_epoch = np.nan

        self.logger.info("---")
        self.logger.info(f"训练完成 | 总耗时: {total_time:.2f}s | 总轮次: {total_epochs}")
        self.logger.info(f"最佳验证AUC: {best_val_auc:.4f} (Epoch {best_val_auc_epoch})")
        self.logger.info(f"最低验证损失: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
        self.logger.info(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 绘制训练历史图表
        self.plot_training_history()

        # 确保最终CSV已保存
        self._save_csv()


# 使用示例（测试用）
if __name__ == "__main__":
    # 简单的示例代码，展示如何使用ModelLogger
    # 注意：运行此示例会创建 ./output_test 目录和文件
    output_test_dir = "./output_test"
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    test_logger = ModelLogger("测试模型", output_test_dir)

    for epoch in range(1, 11):
        test_logger.start_epoch(epoch)

        # 模拟训练过程
        time.sleep(0.1) # 缩短等待时间

        # 模拟训练和验证指标（引入一些随机性）
        train_loss = 1.0 - 0.05 * epoch + 0.05 * np.random.randn()
        train_auc = 0.5 + 0.03 * epoch + 0.02 * np.random.randn()
        val_loss = 1.1 - 0.04 * epoch + 0.06 * np.random.randn()
        val_auc = 0.48 + 0.04 * epoch + 0.03 * np.random.randn()

        # 确保指标在合理范围内
        train_loss = max(0, train_loss)
        train_auc = max(0, min(1, train_auc))
        val_loss = max(0, val_loss)
        val_auc = max(0, min(1, val_auc))

        # 记录结果
        test_logger.end_epoch(epoch, train_loss, train_auc, val_loss, val_auc, learning_rate=0.001 * (0.9 ** epoch))

        # 记录其他指标
        test_logger.log_metrics({'精确率': 0.7 + 0.02 * epoch, '召回率': 0.65 + 0.03 * epoch}, prefix='验证集指标')

    # 完成训练
    test_logger.finish_training()

    print(f"\n示例完成，请查看 {output_test_dir} 目录下的输出文件。")