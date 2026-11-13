"""Biblioteca compartilhada pelos notebooks de analise (Tech Skills Brasil).

Concentra: carga da base, recorte de secoes de beneficios, dicionario de
competencias (Secao 3.4 do artigo), regras de experiencia/formacao, deteccao
de IA generativa e funcoes estatisticas/graficas. Todos os notebooks importam
este modulo para usar exatamente as mesmas definicoes.

Versao dos dados: variavel de ambiente TECHSKILLS_VERSAO
    "tag"   -> vagas_v2026.10.27.parquet (versao fixa publicada; padrao)
    "atual" -> vagas.parquet (base consolidada mais recente)
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import unicodedata

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

BASE = pathlib.Path(__file__).resolve().parent
VERSAO = os.environ.get("TECHSKILLS_VERSAO", "tag")
ARQUIVOS = {
    "tag": BASE / "vagas_v2026.10.27.parquet",
    "atual": BASE / "vagas.parquet",
}
CACHE = BASE / f"cache_processado_{VERSAO}.pkl"
SKILL_GRUPO = json.loads((BASE / "skill_grupo.json").read_text(encoding="utf-8"))

# --------------------------------------------------------------------------
# Agrupamento de areas (Secao 3.3 do artigo)
# --------------------------------------------------------------------------
DEV_DADOS = ["Engenharia de Software", "Backend", "Frontend", "Fullstack", "Mobile",
             "Data", "QA", "DevOps", "Segurança", "Inteligência Artificial"]
SUPORTE_INFRA = ["Suporte Técnico", "Service Desk / Help Desk", "Infraestrutura / Redes",
                 "Field Service / Hardware", "Hardware / Eletrônica"]
G_DEV, G_SUP, G_OUT = "Desenvolvimento e dados", "Suporte e infraestrutura", "Outras"
ORDEM_GRUPOS = [G_DEV, G_SUP, G_OUT]
ORDEM_NIVEIS = ["Estágio", "Aprendiz", "Trainee", "Júnior"]
NIVEIS_PRINCIPAIS = ["Estágio", "Júnior"]  # Aprendiz/Trainee tem poucos casos em Dev e dados


def grupo_da_area(a: str) -> str:
    if a in DEV_DADOS:
        return G_DEV
    if a in SUPORTE_INFRA:
        return G_SUP
    return G_OUT


# --------------------------------------------------------------------------
# Normalizacao e recorte de secoes
# --------------------------------------------------------------------------
def dobrar(s: str) -> str:
    """minusculas e sem acento, preservando o comprimento (indices alinhados)."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in s).lower()


SECOES_DESCARTE = [
    "beneficios", "benefits", "informacoes adicionais", "o que oferecemos",
    "what we offer", "o que voce pode esperar", "what to expect from us",
    "como tornamos seu trabalho", "remuneracao e beneficios", "etapas do processo",
    "sobre o processo seletivo", "e os beneficios", "conheca nossos beneficios",
    "conheca os beneficios", "beneficios para voce", "beneficios oferecidos",
]
SECOES_CONTEUDO = [
    "requisitos", "qualificacoes", "responsabilidades", "principais atividades",
    "atividades", "o que voce vai fazer", "o que voce fara", "o que voce ira fazer",
    "o que esperamos de voce", "o que buscamos", "o que procuramos",
    "o que voce precisa ter ou saber", "esperamos que voce", "e importante que voce",
    "e legal que voce", "sobre a vaga", "para se destacar nessa posicao",
    "para se destacar nesta posicao", "voce ira se diferenciar se tiver",
    "buscamos profissional", "conhecimento em ferramentas", "conhecimentos desejaveis",
    "conhecimentos tecnicos", "desejavel conhecimento", "competencias tecnicas essenciais",
    "competencias tecnicas desejaveis", "o que voce pode esperar do seu dia",
    "requisitos e qualificacoes", "requisitos obrigatorios", "requisitos desejaveis",
    "conhecimentos necessarios", "conhecimentos e habilidades",
    "responsabilidades e atribuicoes", "diferencial", "diferenciais", "idiomas",
    "skills", "what you will do", "what you ll do", "what we are looking for",
    "responsibilities", "requirements", "requirements and qualifications", "qualifications",
]
_RE_DESC = re.compile(r"(?<![a-z])(" + "|".join(sorted(map(re.escape, SECOES_DESCARTE), key=len, reverse=True)) + r")(?![a-z])")
_RE_CONT = re.compile(r"(?<![a-z])(" + "|".join(sorted(map(re.escape, SECOES_CONTEUDO), key=len, reverse=True)) + r")(?![a-z])")


def _eh_titulo(orig: str, ini: int, fim: int) -> bool:
    """Heuristica: titulo = comeca em maiuscula no texto original ou e seguido de ':'."""
    return orig[ini].isupper() or orig[fim:fim + 3].lstrip().startswith(":")


def aparar_beneficios(orig: str) -> str:
    """Remove secoes de beneficios/processo seletivo ate o proximo titulo de conteudo.

    Aproximacao das regras de scraper/rules/secoes_descarte.yml do projeto de
    origem (o texto das descricoes chega achatado, sem quebras de linha).
    """
    f = dobrar(orig)
    eventos = []
    for m in _RE_DESC.finditer(f):
        if _eh_titulo(orig, m.start(), m.end()):
            eventos.append((m.start(), "d"))
    if not eventos:
        return orig
    for m in _RE_CONT.finditer(f):
        if _eh_titulo(orig, m.start(), m.end()):
            eventos.append((m.start(), "c"))
    eventos.sort()
    partes, pos, descartando, ini_desc = [], 0, False, 0
    for p, tipo in eventos:
        if tipo == "d" and not descartando:
            partes.append(orig[pos:p]); descartando, ini_desc = True, p
        elif tipo == "c" and descartando:
            pos, descartando = p, False
    if not descartando:
        partes.append(orig[pos:])
    return "".join(partes)


# --------------------------------------------------------------------------
# Dicionario de competencias (Secao 3.4 do artigo). Regex sobre texto dobrado.
# --------------------------------------------------------------------------
COG, SOC, MET = "Cognitiva", "Socioemocional", "Metacognitiva"
COMPETENCIAS: dict[str, tuple[str, str, str]] = {
    # chave: (dimensao, rotulo, regex)
    "resolucao_problemas": (COG, "Resolução de problemas",
        r"resolu[cç]ao de problema|resolver problema|solu[cç]ao de problema|solucionar problema|problem[- ]solving|resolu[cç]ao de desafios"),
    "pensamento_analitico": (COG, "Pensamento analítico e crítico",
        r"pensamento (analitico|critico|sistemico)|raciocinio (logico|analitico)|capacidade analitica|perfil analitico|visao analitica|senso critico|analise critica|espirito critico|olhar analitico|analitic[oa]\b"),
    "atencao_detalhes": (COG, "Atenção aos detalhes",
        r"aten[cç]ao (aos? )?detalhes?|detalhista|orientad[oa] a detalhes|cuidado com (os )?detalhes|olhar (atento|criterioso)"),
    "comunicacao": (SOC, "Comunicação",
        r"(boa|otima|excelente|clara|eficaz|efetiva|assertiva|forte|bom|boas) (capacidade de )?comunica[cç]ao|comunica[cç]ao (clara|eficaz|efetiva|assertiva|verbal|escrita|interpessoal|oral|objetiva)|(habilidade|capacidade|facilidade|desenvoltura)\w* (de|em|para) (se )?comunica\w*|se comunicar|comunicativ[oa]|comunicar[- ]se"),
    "trabalho_equipe": (SOC, "Trabalho em equipe e colaboração",
        r"trabalho em (equipe|time|grupo)|trabalhar em (equipe|time|grupo)|colaborac(ao|oes)|colaborativ|espirito (de )?(equipe|time)|cooperac|team ?player|teamwork|trabalhar de forma (colaborativa|integrada)"),
    "relacionamento_empatia": (SOC, "Relacionamento interpessoal e empatia",
        r"relacionamento interpessoal|habilidades? interpessoais?|bom relacionamento|empatia|empatic[oa]|inteligencia emocional|sociavel|cordial\w*|cortes\b|educad[oa]\b"),
    "proatividade": (SOC, "Proatividade e iniciativa",
        r"pro-?ativ\w*|\biniciativa\b|protagonis\w*|senso de dono|ownership"),
    "autonomia": (SOC, "Autonomia",
        r"autonom\w*|trabalhar de forma independente|autogerenci\w*|autossuficien\w*"),
    "organizacao_tempo": (SOC, "Organização e gestão do tempo",
        r"gest[ao]o do tempo|gerenciamento do tempo|gerenciar (o )?(seu )?tempo|organizad[oa]s?\b|(boa|otima|excelente|forte|senso de|capacidade de|habilidade de|com) organiza[cç]ao|organiza[cç]ao e (planejamento|disciplina|foco|prioriza\w*|gestao)|prioriza[cr]\w*|cumpri(r|mento de) prazos?|gest[ao]o de prazos"),
    "adaptabilidade": (SOC, "Adaptabilidade e resiliência",
        r"adaptab\w*|se adaptar|adaptar[- ]se|capacidade de adapta\w*|flexibilidade(?! de hor| no hor| de jornada)|resilien\w*|trabalhar sob pressao|lidar com (pressao|mudancas)|ambientes? (dinamicos?|de mudancas)"),
    "comprometimento": (SOC, "Comprometimento e responsabilidade",
        r"comprometimento|comprometid[oa]s?|compromisso com (a |o )?(qualidade|prazos?|resultados?|entregas?)|senso de responsabilidade|assumir responsabilidades?"),
    "orientacao_cliente": (SOC, "Orientação ao cliente",
        r"orienta[cç]ao (ao|para o) cliente|orientad[oa]s? (a|ao|para o) cliente|foco (no|em) cliente|customer (focus|centric|oriented)|satisfacao do cliente|visao de cliente|excelencia no atendimento|bom atendimento|atendimento (de qualidade|humanizado|cordial)|cliente em primeiro"),
    "aprendizagem_continua": (MET, "Aprendizagem contínua e curiosidade",
        r"aprendizado (continuo|constante|rapido)|aprendizagem (continua|constante)|aprender (continuamente|rapidamente|rapido|constantemente)|vontade de aprender|disposicao (para|em) aprender|desejo de aprender|sede de (aprender|conhecimento)|interesse em aprender|facilidade (de|em|para) aprender|capacidade de aprender|curiosidade|curios[oa]s?\b|autodidata|autodesenvolvimento|sempre aprendendo|atualizacao constante|busca (constante|continua) (por|de) (conhecimento|aprendizado|atualizacao|evolucao)|growth mindset"),
}
# Extensao NAO prevista no dicionario do artigo (sinalizada nos notebooks)
EXTENSAO_METACOGNITIVA = {
    "autorregulacao": (MET, "Autorregulação e abertura a feedback (extensão)",
        r"feedbacks?\b|receptiv\w* a (criticas|feedback)|autocritic\w*|autoavalia\w*|autodisciplina|autogestao|reflexao"),
}
DIMENSOES = [COG, SOC, MET]
TECNICA = "Técnica"
IDIOMAS = {"Inglês", "Espanhol"}  # nao contam como competencia tecnica (escolha desta analise)

# --------------------------------------------------------------------------
# Experiencia e formacao
# --------------------------------------------------------------------------
_N = r"(\d{1,2}|um|uma|dois|duas|tres|quatro|cinco|seis)"
EXP_EXCLUI = re.compile(
    r"experiencia (do|de|dos|da) (usuario|usuarios|cliente|clientes|colaborador|colaboradores|candidato|candidatos|consumidor|aluno|uso|compra|trabalho)"
    r"|experiencia (digital|omnichannel)|(ganhar|adquirir|acumular|obter|proporcionar|oferecer|oferecemos|vivenciar|construir|desenvolver|primeira|primeiro|enriquecer|ampliar|agregar|oportunidade de|chance de)\s+(uma\s+|sua\s+|a\s+|de\s+|nova\s+|real\s+|ricas?\s+)?(\w+\s+)?experiencia")
EXP_TEMPO = re.compile(
    _N + r"\s*(\([^)]*\)\s*)?(\+|ou mais)?\s*(anos?|meses|mes)\s*(\w+\s+){0,2}(de|em|na|no|com)?\s*(experiencia|atuacao|vivencia)"
    r"|experiencia\s*(minima\s*)?(de|ha|por|acima de|superior a|maior que)\s*(pelo menos |no minimo |ao menos )?" + _N + r"\s*(anos?|meses|mes)")
EXP_PREVIA = re.compile(
    r"experiencia (previa|anterior|comprovada|minima|profissional|pratica|real|na area|na funcao|no cargo|similar|relevante|no mercado|de mercado)"
    r"|(ter|possuir|tenha|possua|tem|possui|ja ter|comprovar)\s+(alguma\s+|boa\s+|solida\s+|uma\s+)?(\w+\s+)?experiencia")
EXP_TECNOLOGIA = re.compile(r"experiencia (em|na|no|com) ")
EXP_NAO_EXIGE = re.compile(
    r"sem experiencia|nao (e |eh |sera )?(necessari[oa]|exigid[oa]|obrigatori[oa]|requerid[oa])\w*\s*(ter |possuir )?(previa )?(experiencia)"
    r"|nao (precisa|necessita|exige|exigimos|requer|requeremos|e preciso)\s*(de |ter |possuir )?(previa |nenhuma )?experiencia"
    r"|experiencia (previa )?(nao (e|sera) (necessaria|exigida|obrigatoria|requerida)|dispensavel|desnecessaria)"
    r"|nao ha necessidade de experiencia|independente de experiencia|mesmo sem experiencia|ainda que sem experiencia|sem necessidade de experiencia|dispensa experiencia|primeiro emprego")
TEMPO_MAX_MESES = 120  # acima de 10 anos nao e requisito plausivel de vaga de entrada
_EXP_EMPRESA = re.compile(r"(temos|somos|empresa|atuamos|atuando|nossa|nosso|fundad|desde|tradicao|\bha\b)")
_DESEJAVEL_LOCAL = re.compile(r"desejav|diferencia|plus\b|valorizad|nice to have|bonus|sera um|e um")
_DIF_TITULO = re.compile(r"(?<![a-z])(diferenciais?|desejaveis?|conhecimentos desejaveis|nice to have|requisitos desejaveis)(?![a-z])")
_REQ_TITULO = re.compile(r"(?<![a-z])(requisitos|qualificacoes|requisitos obrigatorios|requisitos e qualificacoes|conhecimentos necessarios|responsabilidades|atividades|principais atividades|o que voce vai fazer)(?![a-z])")

FORMACAO = {
    "cursando_superior": r"cursando|graduando|estudante de|matriculad[oa]|a partir do \d+\W{0,2}\s*(periodo|semestre)|\d+\W{0,2}\s*(periodo|semestre)|ensino superior (em andamento|cursando)",
    "superior_completo": r"superior completo|graduacao completa|formad[oa] em|bacharel\w*|graduad[oa]\b|licenciatura|ensino superior",
    "tecnico": r"curso tecnico|ensino tecnico|tecnico em|tecnologo|nivel tecnico",
    "medio": r"ensino medio|2o grau|segundo grau",
    "pos": r"pos[- ]?graduacao|\bmba\b|mestrado|especializacao",
}

# --------------------------------------------------------------------------
# IA generativa (Secao 3.4-b): categorias da taxonomia do projeto
# --------------------------------------------------------------------------
GENAI_SKILLS = {"Copilots & Assistentes de IA", "LLMs & IA Generativa"}
GENAI_EXT = GENAI_SKILLS | {"Agentes de IA & RAG"}  # sensibilidade
GENAI_TERMOS = (r"chat ?gpt|\bgpt[- ]?\d?\b|copilot|claude|cursor\b|windsurf|gemini|\bllms?\b|ia generativa|inteligencia artificial generativa|"
                r"genai|gen ai|prompt\w*|openai|anthropic|lovable|ferramentas? de (ia|inteligencia artificial)|assistentes? de (ia|codigo)|agentes? de (ia|inteligencia)|"
                r"grandes modelos|modelos de linguagem|rag\b|langchain|vibe coding|\bia\b")
_TIT_REQ = re.compile(r"(?<![a-z])(requisitos|qualificacoes|requisitos obrigatorios|requisitos e qualificacoes|conhecimentos necessarios|conhecimentos tecnicos|o que esperamos de voce|o que buscamos|o que procuramos|esperamos que voce|diferenciais?|desejaveis?|conhecimentos desejaveis|para se destacar nessa posicao|voce ira se diferenciar se tiver|e importante que voce|e legal que voce)(?![a-z])")
_TIT_ATV = re.compile(r"(?<![a-z])(responsabilidades|atividades|principais atividades|o que voce vai fazer|o que voce fara|o que voce ira fazer|responsabilidades e atribuicoes|what you will do)(?![a-z])")
_TIT_EMP = re.compile(r"(?<![a-z])(sobre a empresa|quem somos|sobre nos|nossa empresa|sobre a (?!vaga)[a-z]+|nossa missao|nosso proposito|sobre o time|nossos valores)(?![a-z])")
_CTX_EMPRESA = re.compile(r"\b(somos|nossa empresa|nossos clientes|nossas solucoes|a empresa (e|atua|oferece)|utilizamos|usamos|oferecemos|nossa plataforma|nossa tecnologia|nosso time usa)\b")
_CTX_REQ = re.compile(r"(conhecimento|experiencia|familiaridade|dominio|vivencia|nocao|nocoes|habilidade|interesse|desejavel|diferencial|ser[a]? um plus)")
_CTX_VALID = re.compile(r"revis\w+|valid\w+|critic\w+|avali\w+|confer\w+|verific\w+|supervis\w+|checar|qualidade d[oa]s? (codigo|respost)")

# Verbos de execucao x julgamento (exploratorio, objetivo iii)
VERBOS_EXEC = re.compile(r"\b(auxili\w+|apoi\w+|execut\w+|realiz\w+|atend\w+|registr\w+|acompanh\w+|cadastr\w+|alimentar|digit\w+|process\w+|instal\w+|configur\w+|abrir chamados?|monitor\w+|document\w+|suporte a)\b")
VERBOS_JULG = re.compile(r"\b(propor\w*|propondo|sugerir\w*|decidir|defin\w+|planej\w+|projet\w+|avali\w+|diagnostic\w+|investig\w+|otimiz\w+|arquitet\w+|lider\w+|priori\w+|valid\w+|revis\w+|tomada de decisao|tomar decisoes|analis(ar|ando|ando)\b|melhor(ar|ia continua)|estrateg\w+|criar solucoes|desenvolver solucoes)\b")

# --------------------------------------------------------------------------
# Carga e processamento
# --------------------------------------------------------------------------
_COMP_RE = {k: re.compile(v[2]) for k, v in {**COMPETENCIAS, **EXTENSAO_METACOGNITIVA}.items()}
_FORM_RE = {k: re.compile(v) for k, v in FORMACAO.items()}
_GENAI_TERMOS = re.compile(GENAI_TERMOS)


def _tempo_em_meses(m: re.Match) -> float | None:
    g = m.group(0)
    num_txt = re.search(_N, g)
    if not num_txt:
        return None
    t = num_txt.group(1)
    mapa = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6}
    n = int(t) if t.isdigit() else mapa[t]
    return n * 12 if re.search(r"ano", g) else n


def classificar_experiencia(tf: str) -> dict:
    """tf: texto aparado e dobrado. Retorna flags e tempo minimo (meses)."""
    excl = [(m.start(), m.end()) for m in EXP_EXCLUI.finditer(tf)]

    def livre(m):
        return not any(a <= m.start() < b or a < m.end() <= b for a, b in excl)

    def desejavel(pos):
        antes = tf[max(0, pos - 90):pos]
        if _DESEJAVEL_LOCAL.search(antes):
            return True
        d = [x.start() for x in _DIF_TITULO.finditer(tf, 0, pos)]
        r = [x.start() for x in _REQ_TITULO.finditer(tf, 0, pos)]
        return bool(d) and (not r or max(d) > max(r))

    tempos, ocorrencias = [], []
    for nome, rx in (("tempo", EXP_TEMPO), ("previa", EXP_PREVIA), ("tecnologia", EXP_TECNOLOGIA)):
        for m in rx.finditer(tf):
            if not livre(m):
                continue
            if nome == "tempo":
                t = _tempo_em_meses(m)
                # "40 anos de experiencia no mercado" costuma descrever a empresa, nao o candidato
                if t is None or t > TEMPO_MAX_MESES or _EXP_EMPRESA.search(tf[max(0, m.start() - 50):m.start()]):
                    continue
                tempos.append(t)
            ocorrencias.append((nome, m.start(), desejavel(m.start())))
    tipos = {o[0] for o in ocorrencias}
    obrig = [o for o in ocorrencias if not o[2]]
    return {
        "exp_tempo": "tempo" in tipos,
        "exp_previa": "previa" in tipos,
        "exp_tecnologia": "tecnologia" in tipos,
        "exp_qualquer": bool(ocorrencias),
        "exp_obrigatoria": bool(obrig),
        "exp_so_desejavel": bool(ocorrencias) and not obrig,
        "exp_nao_exige": bool(EXP_NAO_EXIGE.search(tf)),
        "exp_meses_min": min(tempos) if tempos else np.nan,
    }


def _pos_primeira_secao_vaga(tf: str, orig: str) -> int:
    """Posicao do primeiro titulo de conteudo (requisitos, atividades...) ou -1."""
    for m in _RE_CONT.finditer(tf):
        if m.group(1) in {"requisitos", "responsabilidades", "atividades", "qualificacoes", "principais atividades",
                          "o que voce vai fazer", "requisitos e qualificacoes", "sobre a vaga", "requirements"} \
                and _eh_titulo(orig, m.start(), m.end()):
            return m.start()
    return -1


def processar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grupo"] = df["area"].astype(str).map(grupo_da_area)
    df["nivel"] = df["seniority"].astype(str)
    df["fonte"] = df["source"].astype(str)
    df["desc"] = df["description"].fillna("")
    df["n_chars"] = df["desc"].str.len()
    df["texto"] = df["desc"].map(aparar_beneficios)
    df["tf"] = df["texto"].map(dobrar)
    df["n_chars_aparado"] = df["texto"].str.len()
    df["data_pub"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["sk"] = df["skills"].fillna("").apply(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    df["sk_tec"] = df["sk"].apply(lambda l: [s for s in l if s not in IDIOMAS])
    df["n_sk_tec"] = df["sk_tec"].str.len()
    df["quartil_tam"] = pd.qcut(df["n_chars_aparado"], 4, labels=["Q1 (curto)", "Q2", "Q3", "Q4 (longo)"])

    pos_sec = [_pos_primeira_secao_vaga(t, o) for t, o in zip(df["tf"], df["texto"])]
    df["pos_secao_vaga"] = pos_sec
    for k, rx in _COMP_RE.items():
        df[f"c_{k}"] = df["tf"].map(lambda t, rx=rx: bool(rx.search(t)))
        # indicador de risco de falso positivo: 1a ocorrencia antes do 1o titulo de requisitos/atividades
        prim = df["tf"].map(lambda t, rx=rx: (rx.search(t).start() if rx.search(t) else -1))
        df[f"pre_{k}"] = (prim >= 0) & (df["pos_secao_vaga"] > 0) & (prim < df["pos_secao_vaga"])
    df["d_tecnica"] = df["n_sk_tec"] > 0
    for dim, nome in ((COG, "cog"), (SOC, "soc"), (MET, "met")):
        cols = [f"c_{k}" for k, v in COMPETENCIAS.items() if v[0] == dim]
        df[f"d_{nome}"] = df[cols].any(axis=1)
    df["n_comp"] = df[[f"c_{k}" for k in COMPETENCIAS]].sum(axis=1)

    exp = pd.DataFrame([classificar_experiencia(t) for t in df["tf"]], index=df.index)
    df = pd.concat([df, exp], axis=1)
    for k, rx in _FORM_RE.items():
        df[f"f_{k}"] = df["tf"].map(lambda t, rx=rx: bool(rx.search(t)))

    df["genai"] = df["sk"].apply(lambda l: any(s in GENAI_SKILLS for s in l))
    df["genai_ext"] = df["sk"].apply(lambda l: any(s in GENAI_EXT for s in l))
    df["sk_copilot"] = df["sk"].apply(lambda l: "Copilots & Assistentes de IA" in l)
    df["sk_llm"] = df["sk"].apply(lambda l: "LLMs & IA Generativa" in l)
    df["v_exec"] = df["tf"].map(lambda t: len(VERBOS_EXEC.findall(t)))
    df["v_julg"] = df["tf"].map(lambda t: len(VERBOS_JULG.findall(t)))
    return df


def carregar(forcar: bool = False) -> pd.DataFrame:
    """Base processada, ja filtrada (descricao >= 200 caracteres, Secao 3.3)."""
    if CACHE.exists() and not forcar:
        return pd.read_pickle(CACHE)
    bruto = pd.read_parquet(ARQUIVOS[VERSAO])
    bruto = bruto[bruto["description"].fillna("").str.len() >= 200].reset_index(drop=True)
    df = processar(bruto)
    df.to_pickle(CACHE)
    return df


def carregar_bruto() -> pd.DataFrame:
    return pd.read_parquet(ARQUIVOS[VERSAO])


# --------------------------------------------------------------------------
# Estatistica e graficos
# --------------------------------------------------------------------------
def prop_ic(x, n, alpha=0.05):
    """proporcao e IC de Wilson (em %)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    lo, hi = proportion_confint(x, n, alpha=alpha, method="wilson")
    return 100 * x / n, 100 * lo, 100 * hi


def tabela_prop(df: pd.DataFrame, col: str, por: list[str]) -> pd.DataFrame:
    g = df.groupby(por, observed=True)[col].agg(["sum", "count"]).reset_index()
    r = g.apply(lambda r: prop_ic(r["sum"], r["count"]), axis=1, result_type="expand")
    g[["pct", "ic_lo", "ic_hi"]] = r
    return g.rename(columns={"sum": "n_sim", "count": "n"})


PALETA = {G_DEV: "#0071E3", G_SUP: "#E8590C", G_OUT: "#7A869A"}
PALETA_NIVEL = {"Estágio": "#2B8A3E", "Aprendiz": "#9C36B5", "Trainee": "#C92A2A", "Júnior": "#1C7ED6"}


def estilo():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.8),
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Text", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "-",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def rotulo_barras(ax, fmt="{:.1f}%", fontsize=8, **kw):
    for c in ax.containers:
        try:
            ax.bar_label(c, fmt=fmt, fontsize=fontsize, padding=2, **kw)
        except Exception:
            pass


def aviso_n(df_tab, n_min=100):
    """lista de celulas com n pequeno (para nota de rodape)."""
    pequenas = df_tab[df_tab["n"] < n_min]
    return pequenas


def padronizar_por_quartil(df, col, trat, quartil="quartil_tam", pesos=None):
    """Prevalencia padronizada (padronizacao direta) por quartil de tamanho do texto.

    trat: coluna booleana de grupo (True = tratado). Retorna (p_trat, p_ctrl) em %.
    pesos: pesos dos quartis; padrao = distribuicao do conjunto total.
    """
    if pesos is None:
        pesos = df[quartil].value_counts(normalize=True)
    res = {}
    for val in (True, False):
        sub = df[df[trat] == val]
        pq = sub.groupby(quartil, observed=True)[col].mean()
        res[val] = float((pq * pesos.reindex(pq.index)).sum() / pesos.reindex(pq.index).sum() * 100)
    return res[True], res[False]


def bootstrap_dif_padronizada(df, col, trat, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    difs = []
    for _ in range(n_boot):
        amostra = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
        if amostra[trat].nunique() < 2:
            continue
        try:
            pt, pc = padronizar_por_quartil(amostra, col, trat)
            difs.append(pt - pc)
        except Exception:
            continue
    return np.percentile(difs, [2.5, 97.5]) if difs else (np.nan, np.nan)


def trecho(orig: str, ini: int, fim: int, janela: int = 140) -> str:
    a, b = max(0, ini - janela), min(len(orig), fim + janela)
    corte = re.sub(r"[\s\u0085\u2028\u2029\ufeff\x00-\x1f\x7f]+", " ", orig[a:b]).strip()  # sem quebras nem caracteres de controle
    return ("…" if a else "") + corte + ("…" if b < len(orig) else "")


def barras_prop(ax, tab, x, hue, ordem_x, ordem_hue, cores, n_min=100, ylim=None):
    """Barras agrupadas com IC de Wilson. Celulas com n < n_min ficam hachuradas e claras."""
    w = 0.8 / len(ordem_hue)
    usados = set()
    topo = 0
    for j, h in enumerate(ordem_hue):
        for i, xv in enumerate(ordem_x):
            r = tab[(tab[x] == xv) & (tab[hue] == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            pouco = r["n"] < n_min
            xpos = i - 0.4 + w * (j + 0.5)
            ax.bar(xpos, r["pct"], w * 0.92, color=cores[h], alpha=0.35 if pouco else 1,
                   hatch="//" if pouco else None, edgecolor="white" if not pouco else cores[h],
                   label=h if h not in usados else None)
            usados.add(h)
            ax.errorbar(xpos, r["pct"], yerr=[[r["pct"] - r["ic_lo"]], [r["ic_hi"] - r["pct"]]],
                        color="#333", capsize=2, lw=1)
            ax.text(xpos, r["ic_hi"] + 0.8, f"{r['pct']:.0f}%\nn={int(r['n'])}", ha="center", va="bottom", fontsize=7)
            topo = max(topo, r["ic_hi"])
    ax.set_xticks(range(len(ordem_x)))
    ax.set_xticklabels(ordem_x)
    ax.set_ylim(0, ylim or topo * 1.22)
    ax.set_ylabel("% dos anúncios")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(ordem_hue), fontsize=8)
    ax.grid(axis="x", visible=False)


def nota_n_pequeno(fig, n_min=100):
    fig.text(0.01, -0.02, f"Barras hachuradas: n < {n_min} (leitura apenas indicativa). Barras de erro: IC 95% de Wilson.",
             fontsize=8, color="#555")


def salvar(fig, nome):
    (BASE / "figuras").mkdir(exist_ok=True)
    fig.savefig(BASE / "figuras" / f"{nome}.png", bbox_inches="tight")


# --------------------------------------------------------------------------
# Conceitos x ferramentas (proposta desta analise; validar com a orientadora)
# --------------------------------------------------------------------------
CONCEITOS = {"Lógica de Programação", "Algoritmos", "Estrutura de Dados", "POO", "Engenharia de Software", "Modelagem de Dados",
             "Design Patterns", "Clean Code", "UML", "DDD", "Estatística", "Qualidade de Software", "Redes de Computadores",
             "Sistemas Operacionais", "Segurança da Informação", "Banco de Dados", "Microserviços", "Machine Learning",
             "Análise de Dados", "Cibersegurança", "Metodologias Ágeis"}
_GENERICOS = {"Banco de Dados", "Linux"}
FERRAMENTAS = {s_ for s_, g in SKILL_GRUPO.items() if g in {"linguagens", "frameworks", "cloud_devops", "bancos"}} - CONCEITOS - _GENERICOS
