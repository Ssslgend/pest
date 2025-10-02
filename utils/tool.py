import datetime
import random
import os
import json
import torch
def generate_filename(type=""):
    today = datetime.datetime.now()
    date_str = today.strftime("%Y%m%d")
    random_number= random.randint(1000,9999)
    return f"results_{type}_{date_str}_{random_number}"   

def save_json(outdir,name,result):
    with open(os.path.join(outdir,name),'w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=4)
    print(f"json 文件已保存！ 路径：{os.path.join(outdir,name)}")

def sava_checkpoint(outdir,model,best_val_auc=0):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # save_name = generate_filename("best_model")
    save_path = "best_model.pth"
    model_save_path = os.path.join(outdir,save_path)
    torch.save(model.state_dict(),model_save_path)
    print(f"🎉 新的最佳模型已保存！Val AUC: {best_val_auc:.6f}，路径: {model_save_path}")