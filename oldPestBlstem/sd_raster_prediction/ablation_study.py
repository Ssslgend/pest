# train_raster_new.py (修改部分)
import argparse
from ablation_config import ABLATION_EXPERIMENTS
import matplotlib as plt
from ablation_config import get_config
from data_processor_raster_new import RasterPredictionDataProcessor
from model.bilstm import BiLSTMModel
import numpy as np

# utils.py (新增函数)
def generate_ablation_report(results, save_path):
    """生成消融实验可视化报告"""
    plt.figure(figsize=(14, 10))
    
    # 1. AUC比较图
    plt.subplot(2, 2, 1)
    experiments = [r['experiment'] for r in results]
    auc_means = [r['avg_metrics']['auc']['mean'] for r in results]
    auc_stds = [r['avg_metrics']['auc']['std'] for r in results]
    
    plt.bar(experiments, auc_means, yerr=auc_stds, capsize=5, color='skyblue')
    plt.title('不同消融配置的AUC比较')
    plt.ylabel('AUC')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.7, 1.0)
    
    # 2. 参数量与AUC关系图
    plt.subplot(2, 2, 2)
    param_counts = [r['avg_parameter_count'] for r in results]
    plt.scatter(param_counts, auc_means, s=100, c='green', alpha=0.7)
    
    # 添加标签
    for i, exp in enumerate(experiments):
        plt.annotate(exp, (param_counts[i], auc_means[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.title('参数量与AUC的关系')
    plt.xlabel('参数量')
    plt.ylabel('AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 训练时间比较
    plt.subplot(2, 2, 3)
    training_times = [r['avg_training_time'] for r in results]
    plt.barh(experiments, training_times, color='salmon')
    plt.title('训练时间比较')
    plt.xlabel('时间 (秒)')
    
    # 4. 组件影响分析
    plt.subplot(2, 2, 4)
    components = ['注意力', '残差', '混合专家', '校准']
    impact_scores = [
        results[0]['avg_metrics']['auc']['mean'] - results[1]['avg_metrics']['auc']['mean'],  # 注意力
        results[0]['avg_metrics']['auc']['mean'] - results[2]['avg_metrics']['auc']['mean'],  # 残差
        results[0]['avg_metrics']['auc']['mean'] - results[3]['avg_metrics']['auc']['mean'],  # MoE
        results[0]['avg_metrics']['auc']['mean'] - results[4]['avg_metrics']['auc']['mean']   # 校准
    ]
    
    plt.bar(components, impact_scores, color=['blue', 'green', 'purple', 'orange'])
    plt.title('组件对AUC的影响')
    plt.ylabel('AUC下降幅度')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 整体布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"消融实验报告保存至: {save_path}")
def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--ablation', type=str, default='full_model',
                        choices=list(ABLATION_EXPERIMENTS.keys()),
                        help='要运行的消融实验名称')
    parser.add_argument('--runs', type=int, default=3,
                        help='每个实验运行次数')
    args = parser.parse_args()
    
    # --- 1. Configuration and Setup --- 
    CONFIG = get_config()
    DEVICE = CONFIG['training']['device']
    
    # 设置消融实验名称
    ablation_name = args.ablation
    ablation_config = ABLATION_EXPERIMENTS[ablation_name]
    print(f"\n=== 开始消融实验: {ablation_config['name']} ===\n")
    
    # 更新模型保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_suffix = f"_{ablation_name}_{timestamp}"
    CONFIG['model_save_path'] = CONFIG['model_save_path'].replace(".pth", f"{ablation_suffix}.pth")
    
    # --- 2. Data Loading and Preparation --- 
    # ... (原有代码不变) ...
    
    # --- 3. Model Initialization --- 
    print("\n--- 模型初始化 (消融配置) --- ")
    model_config = {
        "input_size": input_size,
        "hidden_size": CONFIG['model']['hidden_size'],
        "num_layers": CONFIG['model']['num_layers'],
        "dropout": CONFIG['model']['dropout'],
    }
    # 合并消融配置
    model_config.update(ablation_config['config'])
    
    # 使用消融模型类
    model = AblatedBiLSTMModel(
        config=model_config,
        output_size=CONFIG['model']['output_size']
    ).to(DEVICE)
    
    # 打印模型组件状态
    print(f"模型组件状态:")
    print(f"  注意力机制: {'启用' if model.use_attention else '禁用'}")
    print(f"  残差连接: {'启用' if model.use_residual else '禁用'}")
    print(f"  混合专家系统: {'启用' if model.use_moe else '禁用'}")
    print(f"  概率校准: {'启用' if model.use_calibration else '禁用'}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # --- 4. Loss Function and Optimizer --- 
    # ... (原有代码不变) ...
    
    # --- 5. Training Loop --- 
    # ... (原有代码不变) ...
    
    # --- 6. Final Model Evaluation --- 
    # ... (原有代码不变) ...
    
    # 返回最终评估指标
    return {
        'experiment': ablation_config['name'],
        'ablation_key': ablation_name,
        'metrics': test_metrics,
        'parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_time': total_time
    }

if __name__ == '__main__':
    # 添加消融实验运行器
    if '--run-all' in sys.argv:
        print("\n=== 开始完整消融实验系列 ===")
        
        results = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ablation_results_{timestamp}.csv"
        
        for ablation_key in ABLATION_EXPERIMENTS.keys():
            # 多次运行减少随机性
            run_results = []
            for run in range(args.runs):
                print(f"\n>>> 实验运行 {run+1}/{args.runs}: {ABLATION_EXPERIMENTS[ablation_key]['name']}")
                
                # 修改命令行参数
                sys.argv = [sys.argv[0], '--ablation', ablation_key]
                
                # 运行实验
                result = main()
                run_results.append(result)
            
            # 计算平均指标
            avg_metrics = {}
            for metric in run_results[0]['metrics'].keys():
                values = [r['metrics'][metric] for r in run_results]
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            # 保存汇总结果
            summary = {
                'experiment': ABLATION_EXPERIMENTS[ablation_key]['name'],
                'ablation_key': ablation_key,
                'runs': args.runs,
                'avg_metrics': avg_metrics,
                'avg_parameter_count': np.mean([r['parameter_count'] for r in run_results]),
                'avg_training_time': np.mean([r['training_time'] for r in run_results])
            }
            results.append(summary)
        
        # 保存结果到CSV
        df = pd.DataFrame([
            {
                '实验': r['experiment'],
                'AUC': f"{r['avg_metrics']['auc']['mean']:.4f} ± {r['avg_metrics']['auc']['std']:.4f}",
                'F1分数': f"{r['avg_metrics']['f1']['mean']:.4f} ± {r['avg_metrics']['f1']['std']:.4f}",
                '准确率': f"{r['avg_metrics']['accuracy']['mean']:.4f} ± {r['avg_metrics']['accuracy']['std']:.4f}",
                '参数量': f"{r['avg_parameter_count']:,.0f}",
                '训练时间(秒)': f"{r['avg_training_time']:.1f}"
            }
            for r in results
        ])
        
        df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\n消融实验结果保存至: {results_file}")
        
        # 生成可视化报告
        generate_ablation_report(results, f"ablation_report_{timestamp}.png")
    else:
        main()
# model/bilstm.py
class AblatedBiLSTMModel(nn.Module):
    """支持消融实验的BiLSTM模型"""
    def __init__(self, config, output_size=None):
        super(AblatedBiLSTMModel, self).__init__()
        
        # 基础参数
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.output_size = output_size if output_size is not None else config.get("num_classes", 1)
        
        # 消融实验开关
        self.use_attention = config.get("use_attention", True)
        self.use_residual = config.get("use_residual", True)
        self.use_moe = config.get("use_moe", True)
        self.use_calibration = config.get("use_calibration", True)

        # --- 核心LSTM层 ---
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        lstm_output_size = self.hidden_size * 2
        
        # --- 可选组件 ---
        # 注意力机制
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_size)
        else:
            self.attention = None
            
        # 残差块
        self.residual_blocks = None
        if self.use_residual:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(lstm_output_size, self.dropout)
                for _ in range(min(3, self.num_layers))
            ])
            
        # 混合专家系统
        if self.use_moe:
            self.expert1 = nn.Linear(lstm_output_size, 64)
            self.expert2 = nn.Linear(lstm_output_size, 64)
            self.expert3 = nn.Linear(lstm_output_size, 64)
            self.gate = nn.Linear(lstm_output_size, 3)
            self.final_layer = nn.Linear(64, self.output_size)
        else:
            # 不使用MoE时的简化输出层
            self.final_layer = nn.Linear(lstm_output_size, self.output_size)
            
        # 概率校准
        if self.use_calibration:
            self.prob_calibration = ProbabilityCalibrationLayer(self.output_size, bins=20)
        else:
            self.prob_calibration = None

    def forward(self, x):
        # LSTM基础处理
        output, _ = self.lstm(x)
        
        # 注意力机制
        if self.use_attention and self.attention is not None:
            output = self.attention(output)
        
        # 残差块
        if self.use_residual and self.residual_blocks is not None:
            for block in self.residual_blocks:
                output = block(output)
        
        # 混合专家系统
        if self.use_moe:
            exp1 = F.relu(self.expert1(output))
            exp2 = F.relu(self.expert2(output))
            exp3 = F.relu(self.expert3(output))
            
            gate_scores = F.softmax(self.gate(output), dim=-1)
            gated_output = (gate_scores[:, :, 0:1] * exp1 + 
                          gate_scores[:, :, 1:2] * exp2 + 
                          gate_scores[:, :, 2:3] * exp3)
            output = self.final_layer(gated_output)
        else:
            output = self.final_layer(output)
        
        # 概率校准
        if self.use_calibration and self.prob_calibration is not None:
            output = self.prob_calibration(output)
        
        return output