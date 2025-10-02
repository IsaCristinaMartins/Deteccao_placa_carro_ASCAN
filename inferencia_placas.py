from ultralytics import YOLO
from pathlib import Path
import os, csv, torch

# Entradas/saídas
pasta_saida = r"imagens_frame\quadro"
modelo_peso = r"runs\carplates2\weights\best.pt" 
diretorio_projeto = r"C:\Users\isabel_martins\Documents\ASCAN\Computação Cognitiva\Desafio\runs\detect"
nome_execucao = "predict_python"

# Ajustes recomendados
conf_minima = 0.35         # um pouco mais alto para reduzir FP 
imgsz = 640
device = (0 if torch.cuda.is_available() else "cpu")  # GPU automática 
usar_half = bool(torch.cuda.is_available())          

# Carregar modelo
modelo = YOLO(modelo_peso)

# Predição nos frames coloridos
resultados = modelo.predict(
    source=pasta_saida,
    conf=conf_minima,
    imgsz=imgsz,
    device=device,
    half=usar_half,      # << acelera na GPU e ainda foi lento...
    max_det=3,           
    save=True,           # imagens anotadas
    save_txt=True,       # labels YOLO
    save_conf=True,      
    save_crop=True,      # crops das placas
    project=diretorio_projeto,
    name=nome_execucao,
    exist_ok=True,
    verbose=False
)

# Consolidar CSV com as detecções
dir_saida = os.path.join(diretorio_projeto, nome_execucao)
csv_path = os.path.join(dir_saida, "deteccoes.csv")

# Pasta onde os crops ficam 
crop_dir_base = os.path.join(dir_saida, "crops")
crop_dir_plate = os.path.join(crop_dir_base, "plate")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    writer.writerow(["arquivo", "classe_id", "x1", "y1", "x2", "y2", "conf", "save_dir", "crop_dir"])
    for r in resultados:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        arquivo = Path(r.path).name
        cls = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        xyxy = r.boxes.xyxy.tolist()
        for c, conf, (x1, y1, x2, y2) in zip(cls, confs, xyxy):
            writer.writerow([
                arquivo, int(c), int(x1), int(y1), int(x2), int(y2), float(conf),
                dir_saida, crop_dir_plate  
            ])

print("\n✅ Amém Jesus!!! Deu certo.")
print(f"- Imagens anotadas: {dir_saida}")
print(f"- Crops: {os.path.join(dir_saida, 'crops')}")
print(f"- Labels (txt): {os.path.join(dir_saida, 'labels')}")
print(f"- CSV: {csv_path}")
print(f"- Dispositivo: {device} | half={usar_half} | conf_minima={conf_minima} | imgsz={imgsz}")
#Lembrete: se houver muitos falsos positivos; reduza se estiver perdendo placas.


