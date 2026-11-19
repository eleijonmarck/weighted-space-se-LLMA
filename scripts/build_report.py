import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #0b0b0b; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 17pt; margin-bottom: 4pt; } h2 { font-size: 13pt; margin-top: 16pt; border-bottom: 1px solid #e6e5e0; padding-bottom: 2pt; }
h3 { font-size: 11pt; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 6pt 0; width: 100%; } th, td { border: 1px solid #e6e5e0; padding: 3pt 5pt; text-align: left; }
th { background: #f4f3ef; } code { font-size: 9pt; background: #f4f3ef; padding: 0 2pt; } img { max-width: 100%; }
.pagebreak { page-break-before: always; }
"""


LIST = re.compile(r"^(\s*)([-*]|\d+\.) ")


def normalize(text):
    """Python-Markdown needs a blank line before a list and 4-space indents for nested items."""
    out = []
    for line in text.split("\n"):
        m = LIST.match(line)
        if m:
            indent = len(m.group(1))
            if 0 < indent < 4:
                line = "    " + line.lstrip()
            prev = out[-1] if out else ""
            if prev.strip() and not LIST.match(prev) and not prev.startswith("    "):
                out.append("")
        out.append(line)
    return "\n".join(out)


def md(path):
    text = normalize((ROOT / path).read_text())
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    return html.replace("ᵀ", "<sup>T</sup>")


def main():
    body = md("README.md")
    body += '<div class="pagebreak"></div><h2>Appendix A. Figures</h2>'
    for f, cap in (("layer_sweep.png", "Dev gender AUC after amplifying the article-gender signal in each layer. Selection data only."),
                   ("frontier.png", "Dev-only trade-off between gender AUC gain and Swedish perplexity cost, layers 16 to 23.")):
        if not (ROOT / "figures" / f).exists():
            continue
        body += f'<p><img src="figures/{f}"><br><em>{cap}</em></p>'
    body += '<div class="pagebreak"></div><h2>Appendix B. Design and self-review</h2>' + re.sub(r"<h1>.*?</h1>", "", md("DESIGN.md"), count=1)
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Weight-space editing for Swedish en/ett agreement</title><style>{CSS}</style></head><body>{body}</body></html>"
    (ROOT / "report.html").write_text(html)



def pdf():
    from xhtml2pdf import pisa

    html = (ROOT / "report.html").read_text()
    html = html.replace('src="figures/', f'src="{ROOT}/figures/')
    html = html.replace("max-width: 180mm; margin: 0 auto;", "")
    html = html.replace("<style>", "<style>@page { size: A4; margin: 16mm 15mm; } ")
    with open(ROOT / "report.pdf", "wb") as f:
        ok = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    print("pdf errors:", ok.err)


if __name__ == "__main__":
    main()
    pdf()
