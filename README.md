# Análise de anúncios de vagas de entrada em tecnologia (Tech Skills Brasil)

Notebooks reprodutíveis que analisam **anúncios brasileiros de estágio, aprendiz, trainee e nível júnior em tecnologia**, usando a base aberta do projeto [Tech Skills Brasil](https://github.com/diasgarcia/tech-skills-br).

A análise faz parte de uma pesquisa de iniciação científica sobre **inteligência artificial e o mercado de trabalho júnior em Ciência da Computação**. O foco é descrever como **requisitos de experiência** e **competências técnicas, cognitivas, socioemocionais e metacognitivas** aparecem nos anúncios, e discutir esses padrões à luz da literatura sobre IA generativa.

> **Status: em desenvolvimento.** Os resultados são **preliminares e descritivos**. Parte das classificações ainda depende de validação manual (ver [Limitações](#limitações)).

## O que a análise responde

Cada notebook corresponde a um objetivo específico da pesquisa:

| Notebook | Objetivo | O que faz |
|---|---|---|
| `00_preparacao_amostra` | Preparação | Carrega a base, reproduz os filtros da amostra, compara os números do artigo com as versões publicadas dos dados e documenta o dicionário de competências |
| `01_objetivo_i_experiencia_formacao` | (i) Experiência e formação | Tipos de exigência de experiência, obrigatória × desejável, tempo mínimo, formação declarada |
| `02_objetivo_ii_competencias` | (ii) Competências | Mapas de competências por grupo de área e nível, padronização pelo tamanho do texto, conceitos × ferramentas, diagnóstico de falsos positivos |
| `03_objetivo_iii_combinacoes` | (iii) Combinações | Perfis de cargo, associação entre competências técnicas e socioemocionais, agrupamento de anúncios, verbos de execução × julgamento |
| `04_objetivo_iv_ia_generativa` | (iv) IA generativa | Menções à IA generativa, comparação entre anúncios com e sem menção (padronizada), regressão logística |
| `05_objetivo_v_literatura` | (v) Literatura | Matriz que confronta 15 achados da literatura com o que a amostra mostra e classifica cada comparação |
| `06_robustez` | Robustez | Refaz as estimativas principais sem o LinkedIn, sem anúncios repetidos e sem textos curtos |

Os notebooks já estão **executados**: gráficos e tabelas aparecem ao abri-los, sem precisar rodar nada.

## Estrutura

```text
analise-tech-skills/
├── 00_…06_*.ipynb          notebooks (gerados a partir de _src/)
├── _src/                   fontes dos notebooks (jupytext, formato percent)
├── analise_lib.py          biblioteca compartilhada: carga, recorte de seções,
│                           dicionário de competências, regras de experiência, gráficos
├── executar.py             executa notebooks sem kernel Jupyter e grava as saídas
├── construir.sh            converte _src/ em notebooks e executa
├── skill_grupo.json        mapa habilidade → grupo da taxonomia do projeto de origem
├── tabela_*.csv            tabelas agregadas geradas pelos notebooks
└── requirements.txt
```

## Como rodar

**Requisitos:** Python 3.9 ou superior (testado com 3.9.6).

```bash
git clone https://github.com/leticiagomesilva/ic-tech-skills-analysis.git
cd ic-tech-skills-analysis/analise-tech-skills

# ambiente (o construir.sh espera o venv em .venv)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name tech-skills --display-name "tech-skills (venv)"
```

### Baixar os dados

Os dados **não** estão neste repositório. Baixe-os do [dataset no Hugging Face](https://huggingface.co/datasets/diasgarcia/tech-skills-br) para a pasta `analise-tech-skills/`:

```bash
# versão fixa usada por padrão (tag v2026.10.27)
curl -L -o vagas_v2026.10.27.parquet \
  https://huggingface.co/datasets/diasgarcia/tech-skills-br/resolve/v2026.10.27/vagas.parquet

# opcional: base mais recente (cresce todo dia)
curl -L -o vagas.parquet \
  https://huggingface.co/datasets/diasgarcia/tech-skills-br/resolve/main/vagas.parquet
```

### Executar

```bash
./construir.sh            # gera e executa todos os notebooks (o 00 leva ~30 s na primeira vez)
./construir.sh 04 05      # só os notebooks escolhidos
```

Ou abra os notebooks no Jupyter (`.venv/bin/jupyter lab`) ou no VS Code e escolha o kernel **tech-skills (venv)**.

**Ordem:** execute do `00` ao `04` antes do `05` e do `06`. O `05` lê tabelas geradas pelos anteriores. Alguns arquivos só passam a existir depois de rodar o notebook correspondente (por exemplo, as planilhas de validação do `02` e do `04`).

Para usar a base mais recente em vez da tag fixa:

```bash
TECHSKILLS_VERSAO=atual ./construir.sh
```

> O `executar.py` roda as células em Python puro, sem abrir um kernel Jupyter, e grava gráficos e tabelas no `.ipynb`. Isso é útil em ambientes que bloqueiam conexões locais. Dentro do Jupyter ou do VS Code, os notebooks rodam normalmente com *Run All*.

## Versão dos dados e reprodutibilidade

O padrão é a tag **`v2026.10.27`**: 10.099 anúncios brutos, **9.964** na amostra analítica (descrição com 200 ou mais caracteres). O rascunho do artigo cita um snapshot de 29/10/2026 com 10.317 anúncios, que **não corresponde exatamente a nenhuma versão publicada**; o notebook `00` mostra a comparação. Para citar resultados, fixe e informe a versão dos dados.

## Limitações

- **Corte transversal e descritivo.** Os notebooks não medem mudança no tempo nem efeito da IA. A base cresceu porque a coleta se expandiu, então **meses não devem ser comparados**.
- **Anúncios expressam expectativas de contratação**, não atividades realizadas nem contratações concluídas.
- **Viés da amostra:** cerca de 63% dos anúncios vêm do LinkedIn e cerca de 54% do Sudeste; os termos de busca do projeto de origem moldam o que entra na base.
- **Classificação por regras de texto.** O dicionário de competências, as regras de experiência (inclusive *obrigatória × desejável*) e as listas de verbos são heurísticas. A **precisão do dicionário ainda não foi medida** por leitura humana, e a classificação do papel da IA nos anúncios (requisito, atividade ou contexto) é apenas uma pré-classificação automática.
- **Escolhas a validar:** idiomas fora da dimensão técnica, a divisão entre conceitos e ferramentas, o limite de 10 anos para o tempo mínimo de experiência e o recorte de seções de benefícios, que aproxima as regras do projeto de origem.
- **Padronização por tamanho do texto.** O quartil mais curto tem poucos anúncios com menção à IA, o que torna a padronização frágil nesse ponto.
- O agrupamento por K-means é **exploratório** e tem baixa silhueta; serve para gerar hipóteses.

## Dados e créditos

Os dados vêm do projeto **Tech Skills Brasil**, de Rafael Dias Garcia (código sob licença MIT):

- Repositório: <https://github.com/diasgarcia/tech-skills-br>
- Dataset: <https://huggingface.co/datasets/diasgarcia/tech-skills-br>
- Painel: <https://diasgarcia.github.io/tech-skills-br/>

Ao usar a base, cite o projeto conforme o `CITATION.cff` dele: *Dias Garcia, R. **tech-skills-br — Mapeamento de Skills em Tecnologia no Brasil**, versão 2026.10.27.*

Os anúncios são textos de terceiros coletados em portais de emprego. Este repositório **não redistribui a base**; contém apenas código e tabelas agregadas. As saídas dos notebooks incluem alguns trechos curtos de anúncios, usados como exemplos de conferência.
