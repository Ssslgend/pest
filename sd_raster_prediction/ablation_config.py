# ablation_config.py
ABLATION_EXPERIMENTS = {
    "full_model": {
        "name": "完整模型",
        "config": {
            "use_attention": True,
            "use_residual": True,
            "use_moe": True,
            "use_calibration": True
        }
    },
    "no_attention": {
        "name": "无注意力机制",
        "config": {
            "use_attention": False,
            "use_residual": True,
            "use_moe": True,
            "use_calibration": True
        }
    },
    "no_residual": {
        "name": "无残差连接",
        "config": {
            "use_attention": True,
            "use_residual": False,
            "use_moe": True,
            "use_calibration": True
        }
    },
    "no_moe": {
        "name": "无混合专家系统",
        "config": {
            "use_attention": True,
            "use_residual": True,
            "use_moe": False,
            "use_calibration": True
        }
    },
    "no_calibration": {
        "name": "无概率校准",
        "config": {
            "use_attention": True,
            "use_residual": True,
            "use_moe": True,
            "use_calibration": False
        }
    },
    "base_model": {
        "name": "基础模型",
        "config": {
            "use_attention": False,
            "use_residual": False,
            "use_moe": False,
            "use_calibration": False
        }
    }
}