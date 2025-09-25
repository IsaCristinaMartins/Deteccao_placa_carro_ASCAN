import os 
import cv2



def executar_conversao():
    diretorio_entrada = "imagens_frame/quadro"
    diretorio_saida = "imagens_frame/quadro_cinza"
    os.makedirs(diretorio_saida, exist_ok=True)

    total_convertido = 0
    for arquivo in os.listdir(diretorio_entrada):
        if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            caminho_entrada = os.path.join(diretorio_entrada, arquivo)
            imagem_colorida = cv2.imread(caminho_entrada)
            if imagem_colorida is None:
                continue
            imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
            caminho_saida = os.path.join(diretorio_saida, arquivo)
            cv2.imwrite(caminho_saida, imagem_cinza)
            total_convertido += 1

    print(f"[INFO] Total de arquivos convertidos para cinza: {total_convertido}")

if __name__ == "__main__":
    executar_conversao()

#Converteu tudo, mas continua saindo um valor errado  