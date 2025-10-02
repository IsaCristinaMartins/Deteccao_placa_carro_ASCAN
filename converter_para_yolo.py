import os 
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from pathlib import Path
import random



# Path
images_dir = "data/Placas/images" #C:\Users\isabel_martins\Documents\ASCAN\Computação Cognitiva\Desafio\data\Placas\images
ann_dir = "data/Placas/annotations"
saida_base = "data/Placas/yolo"

# Cria a estrutura YOLO
for p in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(saida_base, p), exist_ok=True)

# Lista de imagens ainda sem conversão
imagens = [f for f in os.listdir (images_dir) if f.lower().endswith((".png"))]
random.shuffle(imagens)

# Divisão para treino vs teste

split = int(0.8 *len(imagens))
treino = imagens[:split]
valid = imagens[split:]

def voc_para_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    linhas = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        
        if cls.lower()!= "license-plate" and cls.lower() != "plate":
            cls = "plate"
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        # YOLO: x_center y_center w h normalizados
        x = (xmin + xmax) / 2 / img_w
        y = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        linhas.append(f"0 {x} {y} {w} {h}")
    return linhas

def processar(lista, subset):
    for nome in tqdm(lista, desc=f"Processando {subset}"):
        img_path = os.path.join(images_dir, nome)
        xml_path = os.path.join(ann_dir, Path(nome).stem + ".xml")
        if not os.path.exists(xml_path):
            continue

        # pega tamanho da imagem
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        linhas = voc_para_yolo(xml_path, w, h)
        if not linhas:
            continue

        # salva imagem e label
        shutil.copy(img_path, os.path.join(saida_base, f"images/{subset}", nome))
        with open(os.path.join(saida_base, f"labels/{subset}", Path(nome).stem + ".txt"), "w") as f:
            f.write("\n".join(linhas))

processar(treino, "train")
processar(valid,  "val")





