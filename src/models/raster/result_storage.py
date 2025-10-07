# result_storage.py
import os
import json
import shutil
import pandas as pd
import numpy as np
import logging
import rasterio
import datetime
import pickle
import sys
import glob
import zipfile
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模块
from sd_raster_prediction.config_future import get_future_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultManager:
    """
    预测结果管理类，用于结构化存储和管理预测结果
    """
    
    def __init__(self, base_dir=None, config=None):
        """
        初始化结果管理器
        
        Args:
            base_dir: 存储基础目录，如果为None则使用配置中的目录
            config: 配置信息，如果为None则从配置文件加载
        """
        # 加载配置
        if config is None:
            self.config = get_future_config()
        else:
            self.config = config
        
        # 确定基础目录
        if base_dir is None:
            self.base_dir = os.path.join('results', 'predictions')
        else:
            self.base_dir = base_dir
        
        # 创建目录结构
        self._create_directory_structure()
        
        # 加载结果索引
        self.index = self._load_index()
        
        logger.info(f"结果管理器初始化完成，基础目录: {self.base_dir}")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        # 主目录
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 子目录
        self.predictions_dir = os.path.join(self.base_dir, 'predictions')
        self.metadata_dir = os.path.join(self.base_dir, 'metadata')
        self.visualizations_dir = os.path.join(self.base_dir, 'visualizations')
        self.analyses_dir = os.path.join(self.base_dir, 'analyses')
        self.exports_dir = os.path.join(self.base_dir, 'exports')
        
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.analyses_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        
        # 索引文件
        self.index_file = os.path.join(self.base_dir, 'prediction_index.json')
    
    def _load_index(self):
        """加载结果索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载索引文件时出错: {e}")
                return {'predictions': [], 'last_id': 0}
        else:
            return {'predictions': [], 'last_id': 0}
    
    def _save_index(self):
        """保存结果索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            logger.info("索引文件已更新")
        except Exception as e:
            logger.error(f"保存索引文件时出错: {e}")
    
    def save_prediction(self, predictions, metadata=None, prediction_config=None, source_dir=None):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典，键为时期名称，值为预测数组
            metadata: 元数据字典，如果为None则从预测结果中提取
            prediction_config: 预测配置，如果为None则使用默认配置
            source_dir: 预测源目录，如果不为None则复制整个目录内容
            
        Returns:
            int: 预测ID
        """
        # 生成新的预测ID
        prediction_id = self.index['last_id'] + 1
        self.index['last_id'] = prediction_id
        
        # 创建时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建预测条目
        prediction_entry = {
            'id': prediction_id,
            'timestamp': timestamp,
            'name': f"prediction_{prediction_id}",
            'periods': list(predictions.keys()) if predictions else [],
            'config': prediction_config
        }
        
        # 创建预测目录
        prediction_dir = os.path.join(self.predictions_dir, f"prediction_{prediction_id}")
        os.makedirs(prediction_dir, exist_ok=True)
        
        # 如果指定了源目录，直接复制整个目录内容
        if source_dir and os.path.exists(source_dir):
            logger.info(f"复制源目录 {source_dir} 到 {prediction_dir}")
            
            # 复制源目录中的文件
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                dest_item = os.path.join(prediction_dir, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
            
            prediction_entry['source_dir'] = source_dir
        
        # 否则，保存预测数组
        else:
            logger.info(f"保存预测结果到 {prediction_dir}")
            
            # 保存每个时期的预测结果
            for period_name, prediction in predictions.items():
                # 确保元数据存在
                if metadata and period_name in metadata:
                    meta = metadata[period_name]
                else:
                    meta = None
                
                # 保存预测数组
                prediction_file = os.path.join(prediction_dir, f"prediction_{period_name}.tif")
                
                if meta:
                    with rasterio.open(prediction_file, 'w', **meta) as dst:
                        dst.write(prediction, 1)
                else:
                    # 如果没有元数据，则只保存为.npy文件
                    np.save(os.path.join(prediction_dir, f"prediction_{period_name}.npy"), prediction)
        
        # 保存元数据
        if metadata:
            metadata_file = os.path.join(self.metadata_dir, f"metadata_{prediction_id}.pkl")
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            prediction_entry['metadata_file'] = metadata_file
        
        # 保存配置
        if prediction_config:
            config_file = os.path.join(self.metadata_dir, f"config_{prediction_id}.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_config, f, ensure_ascii=False, indent=2)
            prediction_entry['config_file'] = config_file
        
        # 添加到索引
        self.index['predictions'].append(prediction_entry)
        self._save_index()
        
        logger.info(f"预测 ID {prediction_id} 已保存")
        return prediction_id
    
    def load_prediction(self, prediction_id):
        """
        加载预测结果
        
        Args:
            prediction_id: 预测ID
            
        Returns:
            tuple: (预测结果字典，元数据字典)
        """
        # 查找预测条目
        prediction_entry = None
        for entry in self.index['predictions']:
            if entry['id'] == prediction_id:
                prediction_entry = entry
                break
        
        if not prediction_entry:
            logger.error(f"未找到预测 ID {prediction_id}")
            return None, None
        
        logger.info(f"加载预测 ID {prediction_id}")
        
        # 预测目录
        prediction_dir = os.path.join(self.predictions_dir, f"prediction_{prediction_id}")
        
        if not os.path.exists(prediction_dir):
            logger.error(f"预测目录 {prediction_dir} 不存在")
            return None, None
        
        # 加载预测结果
        predictions = {}
        metadata = {}
        
        # 查找 TIF 文件
        tif_files = glob.glob(os.path.join(prediction_dir, "*.tif"))
        for tif_file in tif_files:
            filename = os.path.basename(tif_file)
            if filename.startswith("prediction_"):
                period_name = filename.replace("prediction_", "").replace(".tif", "")
                
                with rasterio.open(tif_file) as src:
                    predictions[period_name] = src.read(1)
                    metadata[period_name] = src.meta.copy()
        
        # 如果没有 TIF 文件，查找 NPY 文件
        if not predictions:
            npy_files = glob.glob(os.path.join(prediction_dir, "*.npy"))
            for npy_file in npy_files:
                filename = os.path.basename(npy_file)
                if filename.startswith("prediction_"):
                    period_name = filename.replace("prediction_", "").replace(".npy", "")
                    predictions[period_name] = np.load(npy_file)
        
        # 加载元数据
        if 'metadata_file' in prediction_entry and os.path.exists(prediction_entry['metadata_file']):
            try:
                with open(prediction_entry['metadata_file'], 'rb') as f:
                    metadata = pickle.load(f)
            except Exception as e:
                logger.error(f"加载元数据文件时出错: {e}")
        
        logger.info(f"已加载预测 ID {prediction_id}，包含 {len(predictions)} 个时期")
        return predictions, metadata
    
    def delete_prediction(self, prediction_id):
        """
        删除预测结果
        
        Args:
            prediction_id: 预测ID
            
        Returns:
            bool: 是否成功删除
        """
        # 查找预测条目
        prediction_entry = None
        index = -1
        for i, entry in enumerate(self.index['predictions']):
            if entry['id'] == prediction_id:
                prediction_entry = entry
                index = i
                break
        
        if not prediction_entry:
            logger.error(f"未找到预测 ID {prediction_id}")
            return False
        
        logger.info(f"删除预测 ID {prediction_id}")
        
        # 预测目录
        prediction_dir = os.path.join(self.predictions_dir, f"prediction_{prediction_id}")
        
        # 删除预测目录
        if os.path.exists(prediction_dir):
            shutil.rmtree(prediction_dir)
        
        # 删除元数据文件
        if 'metadata_file' in prediction_entry and os.path.exists(prediction_entry['metadata_file']):
            os.remove(prediction_entry['metadata_file'])
        
        # 删除配置文件
        if 'config_file' in prediction_entry and os.path.exists(prediction_entry['config_file']):
            os.remove(prediction_entry['config_file'])
        
        # 删除可视化文件
        vis_dir = os.path.join(self.visualizations_dir, f"prediction_{prediction_id}")
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)
        
        # 删除分析文件
        analysis_dir = os.path.join(self.analyses_dir, f"prediction_{prediction_id}")
        if os.path.exists(analysis_dir):
            shutil.rmtree(analysis_dir)
        
        # 从索引中移除
        self.index['predictions'].pop(index)
        self._save_index()
        
        logger.info(f"预测 ID {prediction_id} 已删除")
        return True
    
    def list_predictions(self):
        """
        列出所有预测
        
        Returns:
            list: 预测条目列表
        """
        return self.index['predictions']
    
    def export_prediction(self, prediction_id, export_format='zip', include_visualizations=True):
        """
        导出预测结果
        
        Args:
            prediction_id: 预测ID
            export_format: 导出格式，目前支持'zip'
            include_visualizations: 是否包含可视化结果
            
        Returns:
            str: 导出文件路径
        """
        # 查找预测条目
        prediction_entry = None
        for entry in self.index['predictions']:
            if entry['id'] == prediction_id:
                prediction_entry = entry
                break
        
        if not prediction_entry:
            logger.error(f"未找到预测 ID {prediction_id}")
            return None
        
        logger.info(f"导出预测 ID {prediction_id}")
        
        # 预测目录
        prediction_dir = os.path.join(self.predictions_dir, f"prediction_{prediction_id}")
        
        if not os.path.exists(prediction_dir):
            logger.error(f"预测目录 {prediction_dir} 不存在")
            return None
        
        # 创建导出目录
        export_dir = os.path.join(self.exports_dir, f"prediction_{prediction_id}")
        os.makedirs(export_dir, exist_ok=True)
        
        # 导出文件路径
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = os.path.join(self.exports_dir, f"prediction_{prediction_id}_{timestamp}.zip")
        
        # 创建临时导出目录
        temp_export_dir = os.path.join(export_dir, f"temp_{timestamp}")
        os.makedirs(temp_export_dir, exist_ok=True)
        
        # 复制预测文件
        pred_export_dir = os.path.join(temp_export_dir, 'predictions')
        os.makedirs(pred_export_dir, exist_ok=True)
        
        for item in os.listdir(prediction_dir):
            source_item = os.path.join(prediction_dir, item)
            dest_item = os.path.join(pred_export_dir, item)
            
            if os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item)
            else:
                shutil.copy2(source_item, dest_item)
        
        # 复制元数据
        if 'metadata_file' in prediction_entry and os.path.exists(prediction_entry['metadata_file']):
            metadata_dir = os.path.join(temp_export_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)
            
            shutil.copy2(
                prediction_entry['metadata_file'], 
                os.path.join(metadata_dir, os.path.basename(prediction_entry['metadata_file']))
            )
        
        # 复制配置
        if 'config_file' in prediction_entry and os.path.exists(prediction_entry['config_file']):
            config_dir = os.path.join(temp_export_dir, 'config')
            os.makedirs(config_dir, exist_ok=True)
            
            shutil.copy2(
                prediction_entry['config_file'], 
                os.path.join(config_dir, os.path.basename(prediction_entry['config_file']))
            )
        
        # 复制可视化结果
        if include_visualizations:
            vis_dir = os.path.join(self.visualizations_dir, f"prediction_{prediction_id}")
            if os.path.exists(vis_dir):
                vis_export_dir = os.path.join(temp_export_dir, 'visualizations')
                shutil.copytree(vis_dir, vis_export_dir)
        
        # 保存索引信息
        with open(os.path.join(temp_export_dir, 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(prediction_entry, f, ensure_ascii=False, indent=2)
        
        # 创建压缩文件
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_export_dir)
                    zipf.write(file_path, arcname)
        
        # 清理临时目录
        shutil.rmtree(temp_export_dir)
        
        logger.info(f"预测 ID {prediction_id} 已导出到 {export_path}")
        return export_path
    
    def import_prediction(self, import_path):
        """
        导入预测结果
        
        Args:
            import_path: 导入文件路径
            
        Returns:
            int: 预测ID
        """
        if not os.path.exists(import_path):
            logger.error(f"导入文件 {import_path} 不存在")
            return None
        
        logger.info(f"导入预测 {import_path}")
        
        # 创建临时导入目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_import_dir = os.path.join(self.base_dir, f"temp_import_{timestamp}")
        os.makedirs(temp_import_dir, exist_ok=True)
        
        try:
            # 解压文件
            with zipfile.ZipFile(import_path, 'r') as zipf:
                zipf.extractall(temp_import_dir)
            
            # 加载信息文件
            info_file = os.path.join(temp_import_dir, 'info.json')
            if not os.path.exists(info_file):
                logger.error(f"导入文件中没有找到信息文件")
                return None
            
            with open(info_file, 'r', encoding='utf-8') as f:
                import_info = json.load(f)
            
            # 加载预测文件
            predictions_dir = os.path.join(temp_import_dir, 'predictions')
            if not os.path.exists(predictions_dir):
                logger.error(f"导入文件中没有找到预测目录")
                return None
            
            # 加载预测结果
            predictions = {}
            metadata = {}
            
            # 加载 TIF 文件
            tif_files = glob.glob(os.path.join(predictions_dir, "*.tif"))
            for tif_file in tif_files:
                filename = os.path.basename(tif_file)
                if filename.startswith("prediction_"):
                    period_name = filename.replace("prediction_", "").replace(".tif", "")
                    
                    with rasterio.open(tif_file) as src:
                        predictions[period_name] = src.read(1)
                        metadata[period_name] = src.meta.copy()
            
            # 如果没有 TIF 文件，加载 NPY 文件
            if not predictions:
                npy_files = glob.glob(os.path.join(predictions_dir, "*.npy"))
                for npy_file in npy_files:
                    filename = os.path.basename(npy_file)
                    if filename.startswith("prediction_"):
                        period_name = filename.replace("prediction_", "").replace(".npy", "")
                        predictions[period_name] = np.load(npy_file)
            
            # 加载配置
            config_dir = os.path.join(temp_import_dir, 'config')
            config_files = glob.glob(os.path.join(config_dir, "*.json"))
            prediction_config = None
            if config_files:
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    prediction_config = json.load(f)
            
            # 保存预测
            prediction_id = self.save_prediction(
                predictions, 
                metadata=metadata,
                prediction_config=prediction_config
            )
            
            # 导入可视化结果（如果有）
            vis_dir = os.path.join(temp_import_dir, 'visualizations')
            if os.path.exists(vis_dir):
                dest_vis_dir = os.path.join(self.visualizations_dir, f"prediction_{prediction_id}")
                if os.path.exists(dest_vis_dir):
                    shutil.rmtree(dest_vis_dir)
                shutil.copytree(vis_dir, dest_vis_dir)
            
            logger.info(f"预测已导入，ID: {prediction_id}")
            return prediction_id
        
        except Exception as e:
            logger.error(f"导入预测时出错: {e}")
            return None
        
        finally:
            # 清理临时目录
            if os.path.exists(temp_import_dir):
                shutil.rmtree(temp_import_dir)
    
    def find_prediction_by_name(self, name):
        """
        按名称查找预测
        
        Args:
            name: 预测名称
            
        Returns:
            int: 预测ID，如果未找到则返回None
        """
        for entry in self.index['predictions']:
            if entry['name'] == name:
                return entry['id']
        return None
    
    def rename_prediction(self, prediction_id, new_name):
        """
        重命名预测
        
        Args:
            prediction_id: 预测ID
            new_name: 新名称
            
        Returns:
            bool: 是否成功重命名
        """
        # 查找预测条目
        prediction_entry = None
        for entry in self.index['predictions']:
            if entry['id'] == prediction_id:
                prediction_entry = entry
                break
        
        if not prediction_entry:
            logger.error(f"未找到预测 ID {prediction_id}")
            return False
        
        # 更新名称
        prediction_entry['name'] = new_name
        self._save_index()
        
        logger.info(f"预测 ID {prediction_id} 已重命名为 {new_name}")
        return True

def create_result_manager(base_dir=None, config=None):
    """
    创建结果管理器
    
    Args:
        base_dir: 存储基础目录，如果为None则使用配置中的目录
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        ResultManager: 结果管理器实例
    """
    return ResultManager(base_dir=base_dir, config=config)

def save_prediction_results(predictions, metadata=None, config=None, source_dir=None):
    """
    保存预测结果的便捷函数
    
    Args:
        predictions: 预测结果字典，键为时期名称，值为预测数组
        metadata: 元数据字典，如果为None则从预测结果中提取
        config: 配置信息，如果为None则从配置文件加载
        source_dir: 预测源目录，如果不为None则复制整个目录内容
        
    Returns:
        int: 预测ID
    """
    manager = create_result_manager(config=config)
    return manager.save_prediction(
        predictions, 
        metadata=metadata, 
        prediction_config=config,
        source_dir=source_dir
    )

def load_prediction_results(prediction_id=None, name=None, config=None):
    """
    加载预测结果的便捷函数
    
    Args:
        prediction_id: 预测ID
        name: 预测名称，如果prediction_id为None则按名称查找
        config: 配置信息，如果为None则从配置文件加载
        
    Returns:
        tuple: (预测结果字典，元数据字典)
    """
    manager = create_result_manager(config=config)
    
    if prediction_id is None and name is not None:
        prediction_id = manager.find_prediction_by_name(name)
    
    if prediction_id is None:
        logger.error("未指定预测ID或名称")
        return None, None
    
    return manager.load_prediction(prediction_id)

if __name__ == "__main__":
    # 测试结果管理器
    manager = create_result_manager()
    
    # 创建示例预测结果
    predictions = {
        'Month_1': np.random.random((100, 100)),
        'Month_2': np.random.random((100, 100)),
        'Month_3': np.random.random((100, 100))
    }
    
    # 保存预测结果
    prediction_id = manager.save_prediction(predictions, prediction_config={'test': True})
    print(f"已保存预测，ID: {prediction_id}")
    
    # 列出所有预测
    all_predictions = manager.list_predictions()
    print(f"所有预测: {len(all_predictions)}")
    
    # 加载预测结果
    loaded_predictions, loaded_metadata = manager.load_prediction(prediction_id)
    print(f"已加载预测，时期数: {len(loaded_predictions)}")
    
    # 导出预测结果
    export_path = manager.export_prediction(prediction_id)
    print(f"已导出预测到: {export_path}")
    
    # 导入预测结果
    imported_id = manager.import_prediction(export_path)
    print(f"已导入预测，ID: {imported_id}")
    
    # 删除测试预测结果
    manager.delete_prediction(prediction_id)
    manager.delete_prediction(imported_id)
    print("已删除测试预测结果") 