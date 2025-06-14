# DL_hw2
## 必要套件 Requirements
請確保你的環境（Colab 或本地）安裝下列 Python 套件：  

pip install torch torchvision  
pip install torchmetrics  
pip install pycocotools  
pip install tqdm  
pip install numpy  
pip install pillow  
如需語意分割進階功能，可選用：  

pip install segmentation-models-pytorch  

## Eval
請依作業說明將資料集解壓至 data/ 目錄下，結構需如下：  
```
data/  
  mini_voc_seg/  
    test/  
      images/  
      annotations/  
  mini_coco_det/  
    test/  
      images/  
      annotations.json  
  imagenette160/  
    test/  
      images/  
      labels/  
```
``` bash  
python eval.py --weights your_model.pt --data_root data --tasks all  
```

