from pathlib import Path
import numpy as np
import cv2
import pytesseract
import csv
import re
import os


# Caminho da pasta de crops 
PASTA_CROPS = Path(r"C:\Users\isabel_martins\Documents\ASCAN\Computação Cognitiva\Desafio\runs\detect\predict_python\crops\placa")

# Onde salvar o CSV com os resultados
SAIDA_CSV = PASTA_CROPS.parent / "placas_lidas_melhor.csv"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Função para ler imagens com caminho contendo acentos
def imread_unicode(path_like):
    p = Path(path_like)
    data = np.fromfile(str(p), dtype=np.uint8)  # evita problemas de encoding no Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# Pré-processamento: cinza -> unsharp -> equalização -> Otsu -> morf. -> resize  e reza 
def preprocess(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    sharp = cv2.addWeighted(g, 1.8, blur, -0.8, 0)
    eq = cv2.equalizeHist(sharp)
    _, bin_ = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = bin_.shape
    escala = max(1.5, 300 / max(h, 1))  
    bin_ = cv2.resize(bin_, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_LINEAR)
    return bin_

# Regex da placa Mercosul simples 
re_mercosul = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

# Mapa de confusões comuns 
subs_confusas = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8"
})

def corrigir_formato(txt):
    t = re.sub(r"[^A-Z0-9]", "", txt.upper())
    
    if len(t) == 7:
        # Lembrete: letras nas 3 primeiras posições
        for i in range(3):
            if t[i].isdigit():
                if t[i] == "0": t = t[:i] + "O" + t[i+1:]
                if t[i] == "1": t = t[:i] + "I" + t[i+1:]
                if t[i] == "5": t = t[:i] + "S" + t[i+1:]
                if t[i] == "8": t = t[:i] + "B" + t[i+1:]
        # Lembrete: posição 3° deve ser dígito
        if not t[3].isdigit():
            t = t[:3] + t[3].translate(subs_confusas) + t[4:]
        # Lembrete: posições 5° e 6° dígitos
        for i in (5, 6):
            if i < len(t) and not t[i].isdigit():
                t = t[:i] + t[i].translate(subs_confusas) + t[i+1:]
    # Se não bater, tenta substituir confusões globalmente
    if len(t) >= 7 and not re_mercosul.match(t[:7]):
        t2 = t.translate(subs_confusas)
        if len(t2) >= 7 and re_mercosul.match(t2[:7]):
            t = t2
    return t

def ocr_placa(img_bgr):
    bin_ = preprocess(img_bgr)
    cfgs = [
        r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    ]
    candidatos = []
    for cfg in cfgs:
        raw = pytesseract.image_to_string(bin_, config=cfg)
        cand = corrigir_formato(raw)
        if len(cand) >= 6:
            candidatos.append(cand[:7])
    # Preferir quem bate no regex
    for c in candidatos:
        if re_mercosul.match(c):
            return c
    return max(candidatos, key=len) if candidatos else ""

def main():
    if not PASTA_CROPS.exists():
        raise FileNotFoundError(f"Pasta de crops não encontrada: {PASTA_CROPS}")

    imagens = [p for p in PASTA_CROPS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not imagens:
        print("Nenhuma imagem encontrada em:", PASTA_CROPS)
        return

    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "placa_lida"])

        for img_path in imagens:
            img = imread_unicode(img_path)
            if img is None:
                print("Falha ao abrir:", img_path)
                continue
            placa = ocr_placa(img)
            w.writerow([img_path.name, placa])

    print("✅ OCR concluído.")
    print("CSV salvo em:", SAIDA_CSV)

if __name__ == "__main__":
    main()
