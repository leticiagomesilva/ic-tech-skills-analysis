"""Executa notebooks .ipynb SEM kernel Jupyter (nao precisa abrir sockets).

Roda as celulas de codigo em um unico namespace, captura stdout, chamadas a
IPython.display.display (HTML/Markdown) e figuras matplotlib (PNG) e grava tudo
como saidas do proprio .ipynb. Ao abrir o arquivo no Jupyter/VS Code, os
resultados ja aparecem. Uso: python executar.py nome.ipynb [...]
"""
import ast, base64, contextlib, io, os, sys, time, traceback, pathlib, warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output
import IPython.display as ipd


def executar(caminho: str) -> None:
    caminho = pathlib.Path(caminho).resolve()
    os.chdir(caminho.parent)
    sys.path.insert(0, str(caminho.parent))
    warnings.filterwarnings("ignore")
    nb = nbformat.read(caminho, as_version=4)
    ns = {"__name__": "__main__"}
    saidas: list = []
    buf = io.StringIO()

    def esvaziar_texto():
        txt = buf.getvalue()
        if txt:
            saidas.append(new_output("stream", name="stdout", text=txt))
            buf.seek(0); buf.truncate(0)

    def esvaziar_figuras():
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight")
            esvaziar_texto()
            saidas.append(new_output("display_data", data={
                "image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}, metadata={}))
            plt.close(fig)

    def _display(*objs, **kw):
        esvaziar_texto(); esvaziar_figuras()
        for o in objs:
            data = {}
            for mime, meth in (("text/html", "_repr_html_"), ("text/markdown", "_repr_markdown_")):
                if hasattr(o, meth):
                    try:
                        v = getattr(o, meth)()
                    except Exception:
                        v = None
                    if v:
                        data[mime] = v
            data["text/plain"] = repr(o)
            saidas.append(new_output("display_data", data=data, metadata={}))

    ipd.display = _display
    plt.show = lambda *a, **k: esvaziar_figuras()

    n = 0
    t0 = time.time()
    for i, cel in enumerate(nb.cells):
        if cel.cell_type != "code":
            continue
        n += 1
        saidas.clear(); buf.seek(0); buf.truncate(0)
        erro = None
        try:
            arvore = ast.parse(cel.source)
            ultima = arvore.body.pop() if arvore.body and isinstance(arvore.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buf):
                exec(compile(arvore, f"<celula {i}>", "exec"), ns)
                if ultima is not None:
                    valor = eval(compile(ast.Expression(ultima.value), f"<celula {i}>", "eval"), ns)
                    if valor is not None:
                        _display(valor)
        except Exception:
            erro = traceback.format_exc()
        esvaziar_texto(); esvaziar_figuras()
        if erro:
            saidas.append(new_output("error", ename="Erro", evalue="falha na celula", traceback=erro.splitlines()))
        cel.outputs = list(saidas)
        cel.execution_count = n
        if erro:
            nbformat.write(nb, caminho)
            print(f"ERRO na celula {i} de {caminho.name}:\n{erro}")
            sys.exit(1)
    nbformat.write(nb, caminho)
    print(f"{caminho.name}: {n} celulas executadas em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        executar(a)
