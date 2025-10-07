import os
import rasterio
from config_raster_new import get_config

# 获取配置
config = get_config()
feature_map = config['feature_raster_map']

# 尝试打开第一个栅格文件
first_feature = list(feature_map.keys())[0]
raster_path = feature_map[first_feature]

print(f"尝试打开栅格文件: {raster_path}")

try:
    with rasterio.open(raster_path) as src:
        print(f"栅格文件打开成功! 形状: {src.shape}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"NoData值: {src.nodata}")
except Exception as e:
    print(f"打开栅格文件失败: {e}")
    
    # 检查文件是否存在
    if os.path.exists(raster_path):
        print(f"文件存在，但无法打开。可能是文件格式问题或权限问题。")
    else:
        print(f"文件不存在! 检查路径是否正确。")
        
        # 检查目录是否存在
        dir_path = os.path.dirname(raster_path)
        if os.path.exists(dir_path):
            print(f"目录存在，但文件不存在。检查文件名是否正确。")
            
            # 列出目录内容
            files = os.listdir(dir_path)
            print(f"目录 {dir_path} 中的文件:")
            for f in files[:10]:  # 只打印前10个文件
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... 以及其他 {len(files)-10} 个文件")
        else:
            print(f"目录不存在! 检查路径是否正确。") 