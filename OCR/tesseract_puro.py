from pathlib import Path
import cv2
import pytesseract
import csv

# >>> AJUSTE AQUI: pasta com as imagens brutas (frames) <<<
IMAGENS_DIR = Path(r"imagens_frame\quadro")

# (Opcional) aponte para o executável do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def listar_imagens(pasta: Path):
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
    return sorted([p for p in pasta.iterdir() if p.suffix.lower() in EXTS])

def ocr_puro(img_bgr):
    # sem config => Tesseract “padrão” (sem whitelist/psm/oem custom)
    data = pytesseract.image_to_data(img_bgr, config="", output_type=pytesseract.Output.DICT)
    texto = " ".join([w for w in data["text"] if w.strip()])
    confs = [int(c) for c in data["conf"] if c != "-1"]
    conf_media = (sum(confs)/len(confs)) if confs else -1
    return texto, conf_media

def main():
    imagens = listar_imagens(IMAGENS_DIR)
    if not imagens:
        print(f"[AVISO] Nenhuma imagem em {IMAGENS_DIR}")
        return

    saida_csv = IMAGENS_DIR / "baseline_tesseract_puro.csv"
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "texto_bruto", "conf_media"])
        ok = 0
        for imgp in imagens:
            img = cv2.imread(str(imgp))
            if img is None:
                print(f"[AVISO] Não consegui ler: {imgp}")
                continue
            txt, conf = ocr_puro(img)
            w.writerow([imgp.name, txt, f"{conf:.2f}"])
            ok += 1

    print(f"[OK] Processadas {ok} imagens. CSV salvo em: {saida_csv}")

if __name__ == "__main__":
    main()
