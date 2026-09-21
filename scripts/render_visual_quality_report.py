"""Render saved evidence without changing its scores or acceptance thresholds."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def render(directory: Path) -> None:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    cases = report["inpaint"]["cases"]
    summary = report["inpaint"]["summary"]
    names = list(dict.fromkeys(case["asset"] for case in cases))
    chosen = [max((c for c in cases if c["asset"] == name), key=lambda c: c["patch"]["mae_255"]) for name in names]
    sheet = Image.new("RGB", (740, 100 + len(chosen)*200 + 36), "#f2f4f7")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 16), "Bellium - diagnostics de reconstruction", fill="#18212b")
    draw.text((16, 36), "Plus grande erreur par photographie ; tous les cas sont conserves dans report.json.", fill="#364352")
    for i, label in enumerate(("Reference", "Trou", "Candidat Bellium", "Voisin connu")):
        draw.text((16+i*180, 70), label, fill="#18212b")
    for index, case in enumerate(chosen):
        comparison = Image.open(directory / case["comparison"]).convert("RGB")
        width = case["size"][0]
        cx, cy = case["center_xy"]
        y, half, scale = 100+index*200, 11, 7
        for column in range(4):
            crop = comparison.crop((column*width+cx-half, 22+cy-half, column*width+cx+half, 22+cy+half))
            sheet.paste(crop.resize((154, 154), Image.Resampling.NEAREST), (16+column*180, y))
            low, high = (half-case["hole_side"]//2)*scale, (half+case["hole_side"]//2+1)*scale
            draw.rectangle((16+column*180+low-1, y+low-1, 16+column*180+high, y+high), outline="#e3344f")
        status = "refuse" if case["abstained"] else "propose"
        draw.text((16, y+163), f'{case["id"]} : erreur {case["patch"]["mae_255"]:.2f}/255 ; candidat {status}', fill="#18212b")
        score = "non calibree" if case["confidence"] is None else f'{case["confidence"]} (score interne)'
        draw.text((16, y+180), f'Fiabilite chiffree : {score} ; aucune garantie de qualite', fill="#a52237")
    draw.text((16, sheet.height-25), "Sources et licences : voir report.json et les references scikit-image.", fill="#364352")
    sheet.save(directory / "diagnostic-crops.png")
    text = ["# Evaluation visuelle Bellium", "", f'Date UTC : {report["started_at"]}.', "",
            f'**Critere global : {"REUSSI" if report["passed"] else "NON VALIDE"}.**', "",
            f'{summary["cases"]} cas sur {len(names)} photographies ; {summary["accepted"]} resultats acceptes.', "",
            "| Critere | Resultat |", "|---|---|",
            *[f'| {name} | {"Passe" if passed else "Echec"} |' for name, passed in summary["gates"].items()], "",
            f'Erreur moyenne acceptee : {summary["accepted_mean_mae_255"]}.',
            f'Cas acceptes fortement degrades : {summary["severe_accepted_cases"]}.',
            f'Couverture : {summary["coverage"]:.1%}.', "",
            "Les seuils proviennent du protocole fige avant evaluation. Les temps sont descriptifs.",
            "Un score de support n'est pas une probabilite calibree de reconstruction correcte.", "",
            "![Cas diagnostiques](diagnostic-crops.png)", "",
            "## Donnees", "",
            "Sources et licences documentees par [scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).", "",
            *[f'- {a["id"]} : {a["attribution"]}, {a["license"]}, SHA-256 `{a["sha256"]}`.' for a in report["assets"]], "",
            "Le detail, le protocole et les empreintes du moteur sont conserves dans ce dossier."]
    (directory / "RAPPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
