import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, FancyBboxPatch
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# 设置更大的画布，确保内容不会被裁剪
plt.figure(figsize=(14, 10), dpi=300)
ax = plt.gca()
ax.axis('off')

# 布局参数
start_x = 0.1
block_width = 0.15
layer_gap = 0.20
text_offset = 0.02

# 颜色配置
colors = {
    "input": "#3498DB",
    "bilstm": "#E67E22",
    "residual": "#27AE60", 
    "attention": "#E74C3C",
    "moe": "#8E44AD",
    "output": "#2C3E50"
}

def draw_block(ax, x, y, width, height, color, label, alpha=0.7, info=None):
    """绘制基本模块"""
    ax.add_patch(Rectangle((x, y), width, height, 
                          facecolor=color, 
                          edgecolor='black',
                          alpha=alpha))
    plt.text(x + width/2, y + height + text_offset, 
            label, 
            ha='center', va='bottom',
            fontsize=9, 
            fontweight='bold')
    
    if info:
        plt.text(x + width/2, y + height/2, 
            info, 
            ha='center', va='center',
            fontsize=8)

def draw_arrow(ax, start, end, color='gray', linestyle='-', label=None):
    """绘制连接箭头"""
    arrow = FancyArrow(start[0], start[1], 
                      end[0]-start[0], end[1]-start[1],
                      width=0.001, 
                      head_width=0.02,
                      length_includes_head=True,
                      color=color,
                      linestyle=linestyle)
    ax.add_patch(arrow)
    
    if label:
        # 添加数据流说明
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.03
        plt.text(mid_x, mid_y, label, fontsize=8, ha='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# 添加标题（位置上移）
plt.figtext(0.5, 0.94, "American White Moth Pest Prediction Model", 
           ha="center", fontsize=16, weight="bold")
plt.figtext(0.5, 0.91, "BiLSTM with Attention and Mixture of Experts", 
           ha="center", fontsize=12)

# --- INPUT MODULE --- #
# 绘制输入层
input_y = 0.65
draw_block(ax, start_x, input_y, block_width, 0.2, colors["input"], "Input", 
          info="Weather & Geographic\nTime Series\n(batch, seq_len=10, features)")

# --- BILSTM MODULE --- #
# 绘制BiLSTM层
bilstm_x = start_x + layer_gap
bilstm_y = 0.6
draw_block(ax, bilstm_x, bilstm_y, block_width, 0.3, colors["bilstm"], 
          "BiLSTM", info="3 layers, hidden=128\ndropout=0.2\n(batch, seq_len, hidden*2)")

# --- RESIDUAL BLOCKS --- #
# 绘制残差模块
res_x = bilstm_x + layer_gap
res_block_height = 0.1
gap_between_blocks = 0.025
total_res_blocks_height = 3 * res_block_height + 2 * gap_between_blocks
res_start_y = 0.75 - total_res_blocks_height/2

for i in range(3):
    y_pos = res_start_y + i * (res_block_height + gap_between_blocks)
    draw_block(ax, res_x, y_pos, block_width, res_block_height, 
              colors["residual"], f"ResBlock {i+1}", 
              info="LayerNorm + FC\n(batch, seq_len, hidden*2)")

# --- ATTENTION LAYER --- #
# 绘制注意力机制
attn_x = res_x + layer_gap
attn_y = 0.65
draw_block(ax, attn_x, attn_y, block_width, 0.2, colors["attention"],
          "Attention Layer", info="Multi-Head=8\n(batch, hidden*2)")

# --- MIXTURE OF EXPERTS --- #
# 绘制混合专家系统
moe_x = attn_x + layer_gap
moe_width = block_width * 0.9
moe_height = 0.08
expert_gap = 0.04
experts_total_height = 3 * moe_height + 2 * expert_gap
moe_start_y = 0.75 - experts_total_height/2

# 专家模块
expert_infos = [
    "FC + LeakyReLU\n(batch, 64)", 
    "FC + LeakyReLU\n(batch, 64)", 
    "FC + LeakyReLU\n(batch, 64)"
]

expert_ys = []
for i in range(3):
    y_pos = moe_start_y + i * (moe_height + expert_gap)
    expert_ys.append(y_pos)
    draw_block(ax, moe_x, y_pos, moe_width, moe_height,
              colors["moe"], f"Expert {i+1}", info=expert_infos[i])
    
# 绘制门控机制
gate_x = moe_x + moe_width + 0.02
gate_y = 0.65
gate_radius = 0.025
gate_circle = Circle((gate_x, gate_y), 
                    gate_radius, 
                    facecolor='#F1C40F',
                    edgecolor='black')
ax.add_patch(gate_circle)
plt.text(gate_x, gate_y, "G", ha='center', va='center', fontsize=9, fontweight='bold')
plt.text(gate_x, gate_y - 0.04, "Gate", ha='center', va='center', fontsize=8)
plt.text(gate_x, gate_y - 0.06, "(batch, 3)", ha='center', va='center', fontsize=7)

# --- OUTPUT LAYERS --- #
# 绘制概率校准层
calibration_x = moe_x + layer_gap * 0.9
calibration_y = 0.58
draw_block(ax, calibration_x, calibration_y, block_width * 0.8, 0.1, colors["output"],
          "Probability Calibration", info="Non-linear Transform\n(batch, 1)")

# 绘制输出层
output_x = calibration_x
output_y = 0.72
draw_block(ax, output_x, output_y, block_width * 0.8, 0.1, colors["output"],
          "Output Layer", info="Pest Density Prediction\n(batch, 1)")

# --- CONNECTIONS --- #
# 绘制连接线并添加数据流说明
connections = [
    # 输入到BiLSTM
    (start_x + block_width, input_y + 0.1, bilstm_x, bilstm_y + 0.15, 
     "Features\n(batch, 10, features)"),
    
    # BiLSTM到残差块
    (bilstm_x + block_width, bilstm_y + 0.15, res_x, res_start_y + res_block_height * 1.5 + gap_between_blocks, 
     "Hidden States\n(batch, seq_len, hidden*2)"),
    
    # 残差块到注意力
    (res_x + block_width, 0.65, attn_x, attn_y + 0.1, 
     "Enhanced Features"),
    
    # 注意力到专家系统
    (attn_x + block_width, attn_y + 0.1, moe_x, moe_start_y + (moe_height + expert_gap) * 1.5, 
     "Context Vector"),
    
    # 注意力到门控
    (attn_x + block_width*0.9, attn_y + 0.05, gate_x - gate_radius, gate_y,
     ""),
    
    # 门控到输出
    (gate_x + gate_radius, gate_y, output_x, output_y + 0.05,
     "Weighted Features"),
    
    # 注意力到概率校准
    (attn_x + block_width*0.7, attn_y, calibration_x, calibration_y + 0.05,
     "Context Vector"),
]

# 绘制主要连接线
for start_xpos, start_ypos, end_xpos, end_ypos, label in connections:
    draw_arrow(ax, (start_xpos, start_ypos), (end_xpos, end_ypos), label=label)

# 添加专家系统内部连接
for y in expert_ys:
    draw_arrow(ax, 
              (moe_x + moe_width, y + moe_height/2),
              (gate_x - gate_radius, gate_y),
              linestyle='--')

# 添加残差连接
res_block_midpoints = [res_start_y + i * (res_block_height + gap_between_blocks) + res_block_height/2 
                       for i in range(3)]
for i in range(2):
    draw_arrow(ax, 
              (res_x + block_width*0.8, res_block_midpoints[i]),
              (res_x + block_width*0.3, res_block_midpoints[i+1]),
              linestyle=':')

# 添加图例 - 上移到图表上方
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["input"], markersize=10, label='Input Module'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["bilstm"], markersize=10, label='BiLSTM Encoder'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["residual"], markersize=10, label='Residual Block'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["attention"], markersize=10, label='Attention Mechanism'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["moe"], markersize=10, label='Mixture of Experts'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors["output"], markersize=10, label='Output Processing'),
]

legend1 = ax.legend(handles=legend_elements[:3], loc='upper left', bbox_to_anchor=(0.01, 0.2), 
         frameon=True, fontsize=9, title="Components")
ax.add_artist(legend1)

legend2 = ax.legend(handles=legend_elements[3:], loc='upper right', bbox_to_anchor=(0.99, 0.2), 
         frameon=True, fontsize=9, title="Components")
ax.add_artist(legend2)

# 添加连接线图例 - 放在底部
connection_legend_elements = [
    Line2D([0], [0], linestyle='-', color='gray', label='Forward Data Flow'),
    Line2D([0], [0], linestyle='--', color='gray', label='Expert Output'),
    Line2D([0], [0], linestyle=':', color='gray', label='Residual Connection'),
]
legend3 = ax.legend(handles=connection_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.15), 
         ncol=3, frameon=True, fontsize=9, title="Connections")

# 添加简短的底部描述
plt.figtext(0.5, 0.03, 
           "Model processes weather and geographic data to predict American White Moth pest density in Shandong Province", 
           ha='center', va='bottom', fontsize=10, 
           bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.5'))

# 设置适当的边界
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)

# 保存矢量图
plt.savefig('datas/model_architecture.pdf', format='pdf')
plt.savefig('datas/model_architecture.png', dpi=300)
plt.show()