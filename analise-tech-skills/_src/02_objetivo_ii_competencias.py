# %% [markdown]
# # 02 · Objetivo (ii): competências técnicas, cognitivas, socioemocionais e metacognitivas
#
# > *"Mapear as competências técnicas, cognitivas e socioemocionais mencionadas, descrevendo sua frequência por área de atuação e nível de entrada."* (Seção 1.3, objetivo ii)
#
# **Perguntas**
# 1. Quais competências aparecem com mais frequência em cada grupo de área e nível?
# 2. Como o resultado muda quando se controla o **tamanho do texto** (Seção 3.5)?
# 3. Quais habilidades técnicas dominam cada grupo, e qual o peso de **conceitos** frente a **ferramentas**? (Dobslaw et al., 2023)
# 4. Qual o risco de **falsos positivos** do dicionário? Planilha de validação em `validacao_dicionario_amostra.csv` (`[P11]`).
#
# **Ligação com a literatura:** Westby et al. (2026) — comunicação e resolução de problemas em alta nas vagas júnior; Deming (2017) — habilidades sociais; WEF (2025) — pensamento analítico e aprendizagem ao longo da vida; Dobslaw et al. (2023) — conceitos vs. tecnologias.
#
# > Dimensões do artigo: **técnica** (taxonomia do projeto, sem idiomas), **cognitiva**, **socioemocional** e **metacognitiva**. Os valores abaixo são proporções de anúncios que mencionam a competência **pelo menos uma vez**.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, re
import matplotlib.patches as mp
from IPython.display import display, Markdown
import analise_lib as L
L.estilo()
pd.set_option("display.max_colwidth", None, "display.width", 220)
df = L.carregar()
CHAVES = list(L.COMPETENCIAS)
ROT = {k: v[1] for k, v in L.COMPETENCIAS.items()}
DIM = {k: v[0] for k, v in L.COMPETENCIAS.items()}
SIG = {L.COG: "C", L.SOC: "S", L.MET: "M"}
ordem_comp = sorted(CHAVES, key=lambda k: (L.DIMENSOES.index(DIM[k]), -df[f"c_{k}"].mean()))
print(f"{len(df):,} anúncios | {len(CHAVES)} competências no dicionário do artigo")

# %% [markdown]
# ## 1. Mapa de competências por grupo de área
# Cada linha é uma competência, agrupada por dimensão (C = cognitiva, S = socioemocional, M = metacognitiva).

# %%
def matriz(sub_por, colunas):
    m = pd.DataFrame({c: [100 * s[f"c_{k}"].mean() for k in ordem_comp] for c, s in sub_por.items()}, index=ordem_comp)
    m.index = [f"[{SIG[DIM[k]]}] {ROT[k]}" for k in ordem_comp]
    return m

sub = {f"{g}\n(n={(df.grupo == g).sum():,})": df[df.grupo == g] for g in L.ORDEM_GRUPOS}
M = matriz(sub, None)
fig, ax = plt.subplots(figsize=(8.5, 6.2))
sns.heatmap(M, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=60, linewidths=.5, cbar_kws={"label": "% dos anúncios"}, ax=ax)
ax.set_title("Competências mencionadas, por grupo de área"); ax.set_ylabel(""); ax.tick_params(axis="x", labelsize=9)
for y in (np.cumsum([sum(DIM[k] == d for k in ordem_comp) for d in L.DIMENSOES])[:-1]): ax.axhline(y, color="white", lw=4)
L.salvar(fig, "02_mapa_competencias_grupo"); plt.show()

# %% [markdown]
# ## 2. Por grupo **e** nível (Estágio e Júnior)

# %%
sub2 = {}
for g in L.ORDEM_GRUPOS:
    for nv in L.NIVEIS_PRINCIPAIS:
        s = df[(df.grupo == g) & (df.nivel == nv)]
        sub2[f"{g.split(' e ')[0]}\n{nv}\n(n={len(s):,})"] = s
M2 = matriz(sub2, None)
fig, ax = plt.subplots(figsize=(11, 6.2))
sns.heatmap(M2, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=60, linewidths=.5, cbar_kws={"label": "% dos anúncios"}, ax=ax)
ax.set_title("Competências por grupo de área e nível"); ax.set_ylabel(""); ax.tick_params(axis="x", labelsize=8)
for y in (np.cumsum([sum(DIM[k] == d for k in ordem_comp) for d in L.DIMENSOES])[:-1]): ax.axhline(y, color="white", lw=4)
for x in (2, 4): ax.axvline(x, color="white", lw=4)
L.salvar(fig, "02_mapa_competencias_grupo_nivel"); plt.show()

# %% [markdown]
# ## 3. Padronização por quartil de tamanho do texto (Seção 3.5)
#
# Anúncios longos citam mais coisas. A **padronização direta** recalcula a prevalência de cada grupo dando a cada quartil
# de tamanho o **mesmo peso do conjunto total**. Assim, um grupo não "parece" exigir mais só porque seus anúncios são mais longos.
# O gráfico mostra a **diferença entre a proporção padronizada e a bruta** (pontos percentuais).

# %%
pesos = df["quartil_tam"].value_counts(normalize=True)
def prev_pad(s, col):
    pq = s.groupby("quartil_tam", observed=True)[col].mean()
    return 100 * (pq * pesos.reindex(pq.index)).sum() / pesos.reindex(pq.index).sum()

bruta = pd.DataFrame({g: [100 * df[df.grupo == g][f"c_{k}"].mean() for k in ordem_comp] for g in L.ORDEM_GRUPOS}, index=ordem_comp)
pad = pd.DataFrame({g: [prev_pad(df[df.grupo == g], f"c_{k}") for k in ordem_comp] for g in L.ORDEM_GRUPOS}, index=ordem_comp)
dif = (pad - bruta); dif.index = [ROT[k] for k in dif.index]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})
pp = pad.copy(); pp.index = [ROT[k] for k in pp.index]
sns.heatmap(pp, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=60, ax=axes[0], linewidths=.5, cbar=False)
axes[0].set_title("Prevalência padronizada por tamanho do texto (%)"); axes[0].set_xticklabels([g.replace(" e ", "\ne ") for g in L.ORDEM_GRUPOS], fontsize=8)
sns.heatmap(dif, annot=True, fmt="+.1f", cmap="RdBu_r", center=0, vmin=-8, vmax=8, ax=axes[1], linewidths=.5, cbar_kws={"label": "p.p. (padronizada − bruta)"})
axes[1].set_title("O que muda ao padronizar"); axes[1].set_yticklabels([]); axes[1].set_xticklabels([g.replace(" e ", "\ne ") for g in L.ORDEM_GRUPOS], fontsize=8)
L.salvar(fig, "02_padronizacao_tamanho"); plt.show()
print(f"Maior ajuste absoluto: {dif.abs().max().max():.1f} p.p.")

# %% [markdown]
# ## 4. Quatro dimensões do artigo, por grupo (com IC 95%)
# Proporção de anúncios com **ao menos uma** competência de cada dimensão.

# %%
dims = {"d_tecnica": "Técnica", "d_cog": "Cognitiva", "d_soc": "Socioemocional", "d_met": "Metacognitiva"}
cd = {"Técnica": "#0071E3", "Cognitiva": "#E8590C", "Socioemocional": "#2B8A3E", "Metacognitiva": "#9C36B5"}
tabs = []
for c, nome in dims.items():
    t = L.tabela_prop(df, c, ["grupo"]); t["dimensao"] = nome; tabs.append(t)
td = pd.concat(tabs)
fig, ax = plt.subplots(figsize=(11, 4.6))
L.barras_prop(ax, td, "grupo", "dimensao", L.ORDEM_GRUPOS, list(dims.values()), cd)
ax.set_title("Anúncios que mencionam ao menos uma competência de cada dimensão"); ax.set_ylim(0, 115); ax.set_xlabel("")
L.salvar(fig, "02_dimensoes_grupo"); plt.show()

# %% [markdown]
# ## 5. Quantas competências (dicionário) cada anúncio traz?

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for g in L.ORDEM_GRUPOS:
    v = df[df.grupo == g]["n_comp"].clip(upper=8).value_counts(normalize=True).sort_index() * 100
    ax.plot(v.index, v.values, marker="o", color=L.PALETA[g], label=f"{g} (mediana {df[df.grupo == g]['n_comp'].median():.0f})")
ax.set_xlabel("nº de competências distintas (8 = 8 ou mais)"); ax.set_ylabel("% dos anúncios"); ax.legend()
ax.set_title("Distribuição do nº de competências do dicionário por anúncio")
L.salvar(fig, "02_n_competencias"); plt.show()

# %% [markdown]
# ## 6. Competências técnicas (taxonomia do projeto): top 15 por grupo

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, g in zip(axes, L.ORDEM_GRUPOS):
    s = df[df.grupo == g]
    c = s["sk_tec"].explode().value_counts().head(15)[::-1] / len(s) * 100
    ax.barh(c.index, c.values, color=L.PALETA[g])
    for i, v in enumerate(c.values): ax.text(v + .5, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_title(f"{g}\n(n={len(s):,})", fontsize=10); ax.grid(axis="y", visible=False); ax.set_xlabel("% dos anúncios")
fig.suptitle("Habilidades técnicas mais citadas (sem idiomas)", x=.01, ha="left", fontweight="bold")
L.salvar(fig, "02_top_tecnicas"); plt.show()

# %% [markdown]
# ## 7. Conceitos × ferramentas (diálogo com Dobslaw et al., 2023)
#
# Dobslaw et al. observaram que os documentos acadêmicos enfatizam **conceitos** e os anúncios enfatizam **tecnologias**.
# Aqui, a taxonomia do projeto é dividida em duas listas. **A classificação é uma proposta desta análise e precisa ser validada**:
# *conceitos* = fundamentos (lógica, algoritmos, redes, SO...); *ferramentas* = linguagens, frameworks, bancos específicos e ferramentas de nuvem/DevOps.

# %%
CONCEITOS, FERRAMENTAS = L.CONCEITOS, L.FERRAMENTAS   # definidos em analise_lib.py (mesma lista usada no notebook 05)
display(pd.DataFrame({"Conceitos (n=%d)" % len(CONCEITOS): pd.Series(sorted(CONCEITOS)),
                      "Ferramentas (n=%d)" % len(FERRAMENTAS): pd.Series(sorted(FERRAMENTAS))}).fillna("").style.hide(axis="index").set_properties(**{"font-size": "10px", "text-align": "left"}))

df["n_conc"] = df["sk_tec"].apply(lambda l: sum(s in CONCEITOS for s in l))
df["n_ferr"] = df["sk_tec"].apply(lambda l: sum(s in FERRAMENTAS for s in l))
df["tem_conc"] = df["n_conc"] > 0; df["tem_ferr"] = df["n_ferr"] > 0
tc = L.tabela_prop(df, "tem_conc", ["grupo"]).assign(tipo="Conceitos"); tf_ = L.tabela_prop(df, "tem_ferr", ["grupo"]).assign(tipo="Ferramentas")
pd.concat([tc, tf_]).to_csv(L.BASE / "tabela_02_conceito_ferramenta.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4.4))
L.barras_prop(ax, pd.concat([tc, tf_]), "grupo", "tipo", L.ORDEM_GRUPOS, ["Conceitos", "Ferramentas"], {"Conceitos": "#2B8A3E", "Ferramentas": "#C92A2A"})
ax.set_title("Anúncios que citam ao menos um conceito × ao menos uma ferramenta"); ax.set_xlabel("")
L.salvar(fig, "02_conceito_ferramenta"); plt.show()
rt = df.groupby("grupo")[["n_conc", "n_ferr"]].mean().reindex(L.ORDEM_GRUPOS).round(2)
display(rt.rename(columns={"n_conc": "média de conceitos/anúncio", "n_ferr": "média de ferramentas/anúncio"}))

# %% [markdown]
# ## 8. Por área (competências-chave, áreas com n ≥ 50)

# %%
chave = ["comunicacao", "trabalho_equipe", "resolucao_problemas", "proatividade", "aprendizagem_continua", "pensamento_analitico"]
areas = df["area"].astype(str).value_counts(); areas = areas[areas >= 50].index
ma = pd.DataFrame({ROT[k]: [100 * df[df.area == a][f"c_{k}"].mean() for a in areas] for k in chave}, index=[f"{a} (n={(df.area == a).sum()})" for a in areas])
ma = ma.loc[sorted(ma.index, key=lambda x: (L.ORDEM_GRUPOS.index(L.grupo_da_area(x.rsplit(' (n=', 1)[0])), x))]
fig, ax = plt.subplots(figsize=(10, 6.8))
sns.heatmap(ma, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=60, linewidths=.5, cbar_kws={"label": "% dos anúncios"}, ax=ax)
ax.set_title("Competências-chave por área"); ax.tick_params(axis="x", labelsize=8, rotation=20)
L.salvar(fig, "02_competencias_por_area"); plt.show()

# %% [markdown]
# ## 9. Extensão proposta (não prevista no artigo): autorregulação e abertura a feedback
#
# A Seção 2.1 define a dimensão metacognitiva como incluindo "reflexão sobre a própria aprendizagem" e "autorregulação", mas o dicionário
# da Seção 3.4 só tem *aprendizagem contínua e curiosidade*. Se as autoras quiserem manter a definição ampla, falta cobertura.
# Abaixo, um teste exploratório com termos como *feedback, autocrítica, autodisciplina*. **Não entra nos resultados principais.**

# %%
ext = pd.DataFrame({
    "Aprendizagem contínua e curiosidade (no artigo)": [100 * df[df.grupo == g]["c_aprendizagem_continua"].mean() for g in L.ORDEM_GRUPOS],
    "Autorregulação / feedback (extensão)": [100 * df[df.grupo == g]["c_autorregulacao"].mean() for g in L.ORDEM_GRUPOS],
}, index=L.ORDEM_GRUPOS).round(1)
display(ext.style.set_caption("% dos anúncios"))
ex = df[df.c_autorregulacao].sample(6, random_state=3)
display(pd.DataFrame([{"Área": r["area"], "Trecho": L.trecho(r["texto"], m.start(), m.end(), 100)}
                      for _, r in ex.iterrows() for m in [re.search(L.EXTENSAO_METACOGNITIVA["autorregulacao"][2], r["tf"])] if m]).style.hide(axis="index").set_properties(**{"text-align": "left", "font-size": "11px"}))

# %% [markdown]
# ## 10. Risco de falsos positivos do dicionário (diagnóstico)
#
# Um dos riscos citados na Seção 3.6-d é capturar texto **institucional** ("nossa equipe valoriza o trabalho em equipe") em vez de
# um requisito. Um indicador indireto: a proporção de ocorrências **antes do primeiro título de requisitos/atividades**,
# no subconjunto de anúncios onde esse título foi detectado. **Isto não é precisão**, é um alerta para guiar a validação humana.

# %%
tem = df["pos_secao_vaga"] > 0
print(f"Anúncios com título de requisitos/atividades detectado: {tem.mean():.1%} (o diagnóstico vale só para eles)")
rows = []
for k in CHAVES:
    m = df[f"c_{k}"] & tem
    rows.append({"competencia": ROT[k], "dimensao": DIM[k], "n": int(m.sum()), "pct_antes": 100 * df.loc[m, f"pre_{k}"].mean()})
dg = pd.DataFrame(rows).sort_values("pct_antes")
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(dg["competencia"], dg["pct_antes"], color=[cd.get(d, "#888") for d in dg["dimensao"]])
for i, (_, r) in enumerate(dg.iterrows()): ax.text(r["pct_antes"] + .8, i, f"{r['pct_antes']:.0f}% (n={r['n']})", va="center", fontsize=8)
ax.set_xlabel("% das ocorrências antes da seção de requisitos/atividades"); ax.set_title("Alerta de possível contexto institucional"); ax.grid(axis="y", visible=False)
ax.legend(handles=[mp.Patch(color=cd[d], label=d) for d in (L.COG, L.SOC, L.MET)], loc="lower right")
L.salvar(fig, "02_diagnostico_falsos_positivos"); plt.show()

# %% [markdown]
# ## 11. Planilha de validação do dicionário (`[P11]`)
# 15 trechos aleatórios por competência (como descrito na Seção 3.4), prontos para duas leitoras indicarem se **a competência é realmente solicitada ao candidato**.
# Depois de preenchida, a precisão por competência = (nº de "sim") / 15, e o kappa mede a concordância entre as duas leitoras.

# %%
rng = np.random.default_rng(2026)
linhas = []
for k in CHAVES:
    rx = re.compile(L.COMPETENCIAS[k][2])
    pool = df[df[f"c_{k}"]]
    s = pool.sample(min(15, len(pool)), random_state=int(rng.integers(1e6)))
    for _, r in s.iterrows():
        m = rx.search(r["tf"])
        linhas.append({"competencia": ROT[k], "dimensao": DIM[k], "id_anuncio": f"{r['fonte']}:{r['external_id']}", "area": r["area"], "nivel": r["nivel"],
                       "trecho": L.trecho(r["texto"], m.start(), m.end(), 160), "termo_capturado": r["texto"][m.start():m.end()],
                       "ocorre_antes_secao_requisitos": bool(r[f"pre_{k}"]),
                       "solicitada_ao_candidato_leitora1 (sim/nao)": "", "solicitada_ao_candidato_leitora2 (sim/nao)": "", "observacao": ""})
val = pd.DataFrame(linhas)
val.to_csv(L.BASE / "validacao_dicionario_amostra.csv", index=False, encoding="utf-8-sig")
print(f"Planilha salva: validacao_dicionario_amostra.csv ({len(val)} linhas, {val['competencia'].nunique()} competências)")
display(val.groupby("competencia").head(2)[["competencia", "area", "nivel", "termo_capturado", "trecho"]].style.hide(axis="index")
        .set_properties(**{"text-align": "left", "font-size": "11px"}))

# %% [markdown]
# ## 12. Resumo exportável

# %%
res = []
for g in L.ORDEM_GRUPOS:
    for nv in L.NIVEIS_PRINCIPAIS:
        s = df[(df.grupo == g) & (df.nivel == nv)]
        for k in CHAVES:
            x, lo, hi = L.prop_ic(s[f"c_{k}"].sum(), len(s))
            res.append({"grupo": g, "nivel": nv, "competencia": ROT[k], "dimensao": DIM[k], "n": len(s), "pct": round(x, 1), "ic_lo": round(lo, 1), "ic_hi": round(hi, 1)})
pd.DataFrame(res).to_csv(L.BASE / "tabela_02_competencias.csv", index=False)
top = {g: [ROT[k] for k in sorted(CHAVES, key=lambda k: -df[df.grupo == g][f"c_{k}"].mean())[:3]] for g in L.ORDEM_GRUPOS}
Markdown("**Três competências mais frequentes por grupo (dicionário do artigo):**\n\n" + "\n".join(
    f"- {g}: " + ", ".join(f"{c} ({100*df[df.grupo==g]['c_'+[k for k in CHAVES if ROT[k]==c][0]].mean():.0f}%)" for c in top[g]) for g in L.ORDEM_GRUPOS))

# %% [markdown]
# ### Como interpretar
# - Frequências altas de **comunicação** ou **trabalho em equipe** descrevem o que os anúncios *declaram*; não provam que a exigência cresceu ou que decorre da IA.
# - Diferenças entre grupos devem ser lidas **após a padronização por tamanho do texto** (seção 3).
# - Antes de publicar percentuais, é indispensável preencher a planilha de validação (seção 11) e relatar a precisão por competência.
# - A classificação conceito × ferramenta (seção 7) e a exclusão de idiomas da dimensão técnica são **escolhas desta análise**, a validar com a orientadora.
