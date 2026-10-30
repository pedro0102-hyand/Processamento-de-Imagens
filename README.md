# PathoShift Challenge: Entrega 1 (Pathology Image Pipeline)

Pipeline de processamento e classificação de imagens histopatológicas (tiles). Uso exclusivamente experimental, sem finalidade clínica.

> **Dataset provisório:** o professor ainda não divulgou o dataset oficial. Estamos usando o **PathMNIST** (MedMNIST, 64x64, 9 classes de tecido colorretal) só para desenvolver o código. Limiares e conclusões abaixo **devem ser refeitos** com o oficial.

## Status

| Passo | Descrição | Situação |
|---|---|---|
| 1 | Manifesto reprodutível | Feito |
| 2 | Análise exploratória (EDA) | Feito |
| 3A | Rejeição de fundo | Feito |
| 3B | Normalização de cor | Implementado, **uso ainda não decidido** |
| 4 | Baseline de classificação + ablação do pré-processamento | A fazer |
| 5 | Métricas por classe e macro, matriz de confusão, exemplos de erro | A fazer |
| 6 | Registro de versões, seed e tempo de execução | Parcial |

## Como rodar

```bash
pip install -r requirements.txt
python src/download_pathmnist.py --size 64 --per-split 2000   # gera data/{train,val,test}/<classe>/
python src/make_manifest.py --input data --output manifest.jsonl --seed 42
python src/eda.py
python src/background_stats.py      # medidas por imagem -> eda_out/background_measures.csv
python src/background_detector.py   # precisa do passo anterior
python src/color_norm.py
```

`data/`, `manifest.jsonl` e `eda_out/` são gerados e não vão para o Git.

## Scripts

| Arquivo | O que faz |
|---|---|
| `src/download_pathmnist.py` | Baixa o PathMNIST e salva em PNG por split e classe |
| `src/make_manifest.py` | Valida as imagens e gera `manifest.jsonl` (caminho, classe, tamanho, split). Sem split oficial, faz split aleatório provisório (seed 42) |
| `src/eda.py` | Contagem por classe e split, dimensões, cor média, grade de exemplos |
| `src/background_stats.py` | Brilho, desvio padrão do cinza, saturação e IQR por imagem |
| `src/background_detector.py` | Compara 4 regras de detecção de fundo; salva a escolhida em `configs/background_rule.json` |
| `src/color_norm.py` | Normalização de cor (Reinhard por tile, em LAB); salva `configs/color_norm.json` |

## O que encontramos (PathMNIST, validação)

- 6000 imagens usadas (2000 por split), todas 64x64 RGB. O **teste tem proporção de classes diferente** do treino, o que é coerente com ele vir de outra coleta.
- **Cor média** separa só `adipose` e `mucus` (tecidos pálidos). As outras seis classes são parecidas, então o baseline precisa de **textura**, não só de cor.
- **Fundo:** a classe `background` tem dois aspectos (roxo escuro e rosa muito claro). A melhor regra foi `desvio_cinza < 10,5` ou `brilho < 105` (precisão 0,91, recall 0,86, F1 0,88). A medida IQR **não melhorou** a detecção. Cerca de 14% do fundo tem textura e escapa. Alguns `debris` são rejeitados por engano.
- **Normalização de cor:** corrige o violeta vivo de um grupo de tiles `debris`, mas gera **halos verdes/ciano** em áreas brancas. Como força a cor média de todos os tiles para a mesma, pode apagar diferenças entre classes. Só a ablação do passo 4 decide se entra no pipeline.
- 64x64 é pequeno demais para avaliar foco e artefatos de aquisição.

## Pendências

1. Baseline: features de cor e textura (ex.: histogramas HSV/LAB, LBP) + classificador clássico.
2. Ablação no mesmo baseline: sem pré-processamento, com rejeição de fundo, com normalização de cor (`ab` e `full`).
3. Avaliação final: precisão, recall e F1 por classe, F1 macro, matriz de confusão, exemplos de erro. **O teste só é usado uma vez, no final.**
4. Registro: `pip freeze`, tempo de execução e configuração de cada experimento.
5. Trocar para o dataset oficial, adaptar ao formato de submissão (a definir) e refazer limiares.

## Regras que seguimos

- Parâmetros escolhidos **só no treino**; validação só avalia; teste fora de qualquer escolha.
- Seed fixa (42).
- Sem ID de paciente/lâmina no PathMNIST, o split aleatório pode ter vazamento. Com o dataset oficial, usar o split deles.