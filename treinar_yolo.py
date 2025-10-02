from ultralytics import YOLO
from pathlib import Path
import torch, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/Placas/yolo/placa.yaml")
    ap.add_argument("--model", default="yolov8n.pt")  # n/s/m conforme sua GPU
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--workers",type=int, default=8)
    ap.add_argument("--project",default="runs")
    ap.add_argument("--name",   default="carplates")
    ap.add_argument("--device", default=("0" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    # Treino
    modelo = YOLO(args.model)
    res_treino = modelo.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        workers=args.workers, project=args.project, name=args.name, device=args.device
    )

    # Descobre caminho do best.pt
    best_path = None
    if hasattr(modelo, "trainer") and getattr(modelo.trainer, "best", None):
        best_path = Path(modelo.trainer.best)
    elif hasattr(res_treino, "save_dir"):
        best_path = Path(res_treino.save_dir) / "weights" / "best.pt"

    if best_path and best_path.exists():
        print(f"[OK] Treino concluído. Pesos: {best_path}")
        modelo = YOLO(str(best_path))
    else:
        print("[AVISO] Não encontrei best.pt; validando com o modelo atual.")

    # Validação
    res_val = modelo.val(
        data=args.data, imgsz=args.imgsz, workers=args.workers,
        project=args.project, name=f"{args.name}_val", device=args.device
    )

    
    print(f"[VAL] mAP50: {getattr(res_val, 'metrics', {}).get('map50', 'N/A')}")
    print(f"[VAL] Pasta de validação: {getattr(res_val, 'save_dir', 'runs/')}")

if __name__ == "__main__":
    main()
