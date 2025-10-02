# Reconhecimento e Visualização de Placas Automotivas

![Visualização de Placa](https://sensoreng.com.br/wp-content/uploads/2021/03/lpr-leitura-de-placas-de-veiculos-monitoramento-de-portaria-online-armanzenamento-em-nuvem-300x169.jpg)

## Objetivo
Este projeto tem como objetivo estudar e implementar um sistema de **detecção e reconhecimento automático de placas de veículos (OCR)**, utilizando visão computacional e técnicas de aprendizado de máquina, como requisito parcial para a formatura no programa de aperfeiçoamento de estagiários do Instituto Atlântico. 

## Estrutura Inicial do Projeto
A ideia é que ao longo do desenvolvimento o projeto seja mais detalhado, mas os passos gerais serão:

1. **Coleta de Dados**
   - Vá em 'entendendo_processamentos.md' para entender quantos frames você irá precisar no seu processo. 
   - Criação e organização do banco de imagens de placas de veículos (dataset local).
   - Separação em pastas: `train/`, `test/`, `validation/`.

2. **Pré-processamento**
   - Conversão e padronização das imagens.
   - Redimensionamento e normalização.
   - Extração de regiões de interesse (ROI) das placas.

3. **Visualização Inicial**
   - Exibição de exemplos de placas originais e pré-processadas.
   - Criação de organogramas para entender o fluxo de processamento.

4. **Detecção da Placa**
   - Utilização de técnicas de visão computacional (OpenCV, Haar Cascades, YOLO, etc.).
   - Valores bons/aceitáveis para um detector simples de placas:
         - mAP50 (mean Average Precision em IoU=0.5) → ≥0,7 já é bom, ≥0,85 ótimo
         - mAP50-95 (mais rigoroso) → ≥0,5 aceitável, ≥0,65 bom
         - Precision → ≥0,8 bom (quer dizer poucos falsos positivos)
         - Recall → ≥0,7 bom (não perder muitas placas)
         - Loss (box/class) → valores caindo e estabilizando, ideal <1 no final




   - Delimitação da placa dentro da imagem.

5. **Reconhecimento de Caracteres (OCR)**
   - Aplicação de bibliotecas como **Tesseract OCR**.
   - Teste com diferentes abordagens de pré-processamento para melhorar acurácia.

6. **Treinamento de Modelos**
   - Construção de modelos de classificação e reconhecimento.
   - Avaliação de métricas (precisão, recall, F1-score).

7. **Visualização dos Resultados**
   - Exibição dos resultados em imagens com as placas detectadas e reconhecidas.
   - Gráficos comparando diferentes abordagens.


## Estrutura do Banco de Imagens (Organograma)

![Organograma Banco de Imagens](./Desafio/imagens_README/organograma_readme_.png)

## Observação
Este README será atualizado à medida que o projeto for evoluindo, documentando cada etapa com exemplos práticos, imagens e métricas de desempenho.


