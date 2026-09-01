"""Génère le rapport final professionnel du travail HERGI au format PDF."""

import json
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_sarima"
OUT = RESULTS / "rapport_final_hergi"
PDF = OUT / "Rapport_final_HERGI_ARIMA_SARIMA_SARIMAX.pdf"

BLUE = "#123B5D"
CYAN = "#1786A6"
RED = "#C43B3B"
GOLD = "#D89B2B"
INK = "#1E2933"
GREY = "#64727E"
LIGHT = "#EEF4F7"
WHITE = "#FFFFFF"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


SARIMA = load_json(RESULTS / "all_9892_best_model/evaluation_summary.json")
SARIMAX = load_json(RESULTS / "sarimax_all_9892_compact/evaluation_summary.json")
FORECAST_S = load_json(RESULTS / "predictions_2000_2025/report/numeric_summary.json")
VALID_S = load_json(RESULTS / "predictions_2000_2025/validation_summary.json")
VALID_X = load_json(RESULTS / "sarimax_predictions_2000_2025/validation_summary.json")


def base_page(title, subtitle=None, section=None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, .955), 1, .045, color=BLUE, transform=ax.transAxes))
    if section:
        ax.text(.055, .975, section.upper(), color=WHITE, fontsize=8, weight="bold", va="center", transform=ax.transAxes)
    ax.text(.055, .925, title, color=BLUE, fontsize=20, weight="bold", va="top", transform=ax.transAxes)
    if subtitle:
        ax.text(.055, .89, subtitle, color=GREY, fontsize=9.5, va="top", transform=ax.transAxes)
    ax.plot([.055, .945], [.865, .865], color=CYAN, lw=1.2, transform=ax.transAxes)
    return fig, ax


def footer(fig, page):
    ax = fig.axes[0]
    ax.plot([.055, .945], [.035, .035], color="#D7E1E7", lw=.8, transform=ax.transAxes)
    ax.text(.055, .018, "Projet Climate — Rapport final HERGI", color=GREY, fontsize=7.5, transform=ax.transAxes)
    ax.text(.945, .018, str(page), color=GREY, fontsize=7.5, ha="right", transform=ax.transAxes)


def para(ax, text, x, y, width=98, size=9.4, color=INK, weight="normal", spacing=1.35):
    wrapped = "\n".join(textwrap.fill(p, width=width) for p in text.split("\n"))
    ax.text(x, y, wrapped, fontsize=size, color=color, weight=weight, va="top", linespacing=spacing, transform=ax.transAxes)
    return y - (wrapped.count("\n") + 1) * size / 760 * spacing


def section_label(ax, text, y):
    ax.text(.06, y, text.upper(), color=CYAN, fontsize=9, weight="bold", va="top", transform=ax.transAxes)
    return y - .028


def metric_card(ax, x, y, w, h, value, label, note="", color=CYAN):
    ax.add_patch(FancyBboxPatch((x, y-h), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", facecolor=LIGHT, edgecolor="#D6E3EA", lw=.8, transform=ax.transAxes))
    ax.text(x+.018, y-.025, value, fontsize=19, weight="bold", color=color, va="top", transform=ax.transAxes)
    ax.text(x+.018, y-.068, label, fontsize=8.5, weight="bold", color=INK, va="top", transform=ax.transAxes)
    if note:
        ax.text(x+.018, y-.095, textwrap.fill(note, 30), fontsize=7.2, color=GREY, va="top", transform=ax.transAxes)


def add_image(fig, path, rect, title=None):
    ax = fig.add_axes(rect)
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, weight="bold", color=INK, pad=4)
    return ax


def bullet_list(ax, items, x, y, width=92, size=9.2, gap=.012):
    for item in items:
        wrapped = textwrap.wrap(item, width=width)
        ax.text(x, y, "•", color=CYAN, fontsize=size+2, va="top", transform=ax.transAxes)
        ax.text(x+.022, y, "\n".join(wrapped), color=INK, fontsize=size, va="top", linespacing=1.35, transform=ax.transAxes)
        y -= len(wrapped) * size / 720 * 1.35 + gap
    return y


def draw_table(fig, rect, headers, rows, widths=None, font_size=8.2):
    ax = fig.add_axes(rect); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center", colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(font_size); table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D5E0E6"); cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor(BLUE); cell.get_text().set_color(WHITE); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if r % 2 else LIGHT)
            cell.get_text().set_color(INK)
    return ax


def monthly_aggregate(path):
    total = {}
    for chunk in pd.read_csv(path, usecols=["DATE", "T_PRED", "CI95_LOW", "CI95_HIGH"], chunksize=250000):
        chunk["WIDTH"] = chunk.CI95_HIGH - chunk.CI95_LOW
        g = chunk.groupby("DATE").agg(sum_t=("T_PRED", "sum"), n=("T_PRED", "size"), sum_w=("WIDTH", "sum"))
        for idx, row in g.iterrows():
            acc = total.setdefault(int(idx), [0., 0, 0.])
            acc[0] += float(row.sum_t); acc[1] += int(row.n); acc[2] += float(row.sum_w)
    frame = pd.DataFrame([(k, v[0]/v[1], v[2]/v[1]) for k, v in total.items()], columns=["DATE", "T", "WIDTH"]).sort_values("DATE")
    frame["DT"] = pd.to_datetime(frame.DATE.astype(str), format="%Y%m")
    return frame


def cover(pdf, page):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), .095, 1, color=BLUE, transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((.095, .72), .905, .28, color=LIGHT, transform=ax.transAxes))
    ax.text(.15, .90, "RAPPORT FINAL", fontsize=13, weight="bold", color=CYAN, transform=ax.transAxes)
    ax.text(.15, .82, "Modélisation et prévision\nde la température", fontsize=31, weight="bold", color=BLUE, va="top", linespacing=1.05, transform=ax.transAxes)
    ax.text(.15, .69, "ARIMA • SARIMA • SARIMAX", fontsize=17, weight="bold", color=RED, transform=ax.transAxes)
    ax.text(.15, .635, "Analyse mensuelle sur 9 892 points spatiaux\nPrévisions 2000–2025", fontsize=13, color=INK, linespacing=1.5, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((.15, .40), .72, .15, boxstyle="round,pad=.018,rounding_size=.018", facecolor=BLUE, edgecolor=BLUE, transform=ax.transAxes))
    ax.text(.18, .515, "RÉSULTAT PRINCIPAL", color="#A9DCE9", fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(.18, .47, "SARIMAX : RMSE 1,278 °C", color=WHITE, fontsize=21, weight="bold", transform=ax.transAxes)
    ax.text(.18, .43, "Gain de 11,5 % face à SARIMA sur la comparaison équitable", color=WHITE, fontsize=10, transform=ax.transAxes)
    ax.text(.15, .255, "Périmètre", color=CYAN, fontsize=9, weight="bold", transform=ax.transAxes)
    ax.text(.15, .215, "Historique 1960–1999  |  Backtest 1996–1999  |  Horizon 2000–2025", color=INK, fontsize=10, transform=ax.transAxes)
    ax.text(.15, .13, "Équipe projet Climate — Travail HERGI", color=BLUE, fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(.15, .095, f"Version finale • {date.today().strftime('%d/%m/%Y')}", color=GREY, fontsize=9, transform=ax.transAxes)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def executive(pdf, page):
    fig, ax = base_page("Synthèse exécutive", "Les conclusions essentielles en une page", "01 — Synthèse")
    metric_card(ax, .06, .82, .2, .14, "9 892", "points spatiaux", "Couverture complète de la grille")
    metric_card(ax, .285, .82, .2, .14, "1,445 °C", "RMSE SARIMA", "Backtest hors échantillon", BLUE)
    metric_card(ax, .51, .82, .2, .14, "1,278 °C", "RMSE SARIMAX", "ETP + SWI", RED)
    metric_card(ax, .735, .82, .2, .14, "11,5 %", "gain RMSE", "SARIMAX face à SARIMA", GOLD)
    y = section_label(ax, "Verdict", .625)
    y = para(ax, "Le travail HERGI établit une chaîne complète et reproductible : préparation mensuelle, sélection des ordres, backtest temporel, estimation aux 9 892 points, sauvegarde compacte des paramètres et production de 3 086 304 prévisions mensuelles par famille pour 2000–2025.", .06, y, 105, 10)
    y -= .025
    y = bullet_list(ax, [
        "SARIMA est une référence statistique solide : RMSE moyen 1,445 °C, MAE 1,166 °C et gain médian de 30,4 % face au naïf saisonnier.",
        "SARIMAX est le meilleur modèle mensuel évalué à périmètre identique : RMSE 1,278 °C, MAE 1,016 °C et victoire sur 95,8 % des points face à SARIMA.",
        "ETP est la variable exogène dominante : coefficient standardisé moyen 3,703, positif sur 100 % des points. SWI apporte un signal plus faible et spatialement variable.",
        "Les prévisions 2000–2025 sont techniquement complètes et sans valeur manquante. Elles restent des extrapolations statistiques, non une validation climatique récente.",
    ], .06, y, 100, 9.4, .018)
    ax.add_patch(FancyBboxPatch((.06, .12), .88, .12, boxstyle="round,pad=.015,rounding_size=.012", facecolor="#FFF6E3", edgecolor="#F0D59A", transform=ax.transAxes))
    ax.text(.085, .215, "POINT DE VIGILANCE", fontsize=8.5, weight="bold", color="#9A6410", transform=ax.transAxes)
    para(ax, "On peut défendre la qualité de SARIMAX face à SARIMA et au benchmark saisonnier dans ce protocole. On ne peut pas affirmer une supériorité sur tous les modèles du marché sans test strictement identique sur les mêmes dates, points, variables et horizons.", .085, .185, 100, 9, INK)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def data_method(pdf, page):
    fig, ax = base_page("Données et protocole expérimental", "Une séparation temporelle conçue pour éviter la fuite d’information", "02 — Méthodologie")
    y = section_label(ax, "Jeu de données", .83)
    y = bullet_list(ax, [
        "Source traitée : SAFRAN mensuel nettoyé, avec coordonnées LAMBX/LAMBY.",
        "Cible : température mensuelle T. Variables exogènes SARIMAX : ETP et SWI.",
        "Volume : 4 748 160 observations, 9 892 points et 480 mois de janvier 1960 à décembre 1999.",
        "Aucune observation réelle 2000–2025 n'est présente dans le jeu local utilisé pour le rapport.",
    ], .06, y, 100, 9.4, .016)
    y -= .02
    y = section_label(ax, "Chronologie", y)
    # Timeline
    ax.plot([.10, .90], [y-.07, y-.07], color="#B7C8D2", lw=6, solid_capstyle="round", transform=ax.transAxes)
    segments = [(.10, .58, "1960–1995", "Apprentissage\ncomparatif", BLUE), (.58, .68, "1996–1999", "Backtest\nhors échantillon", RED), (.68, .90, "2000–2025", "Prévisions\nfinales", GOLD)]
    for x1, x2, years, label, color in segments:
        ax.plot([x1, x2], [y-.07, y-.07], color=color, lw=8, solid_capstyle="butt", transform=ax.transAxes)
        ax.text((x1+x2)/2, y-.025, years, ha="center", fontsize=9, weight="bold", color=color, transform=ax.transAxes)
        ax.text((x1+x2)/2, y-.115, label, ha="center", va="top", fontsize=8, color=INK, transform=ax.transAxes)
    y -= .20
    y = section_label(ax, "Chaîne finale de prévision", y)
    labels = [("Historique", "T, ETP, SWI\n≤ 1999"), ("Estimation", "Paramètres\ncompacts"), ("Rechargement", "État appris\nreconstitué"), ("Prévision", "312 mois\n2000–2025")]
    xs = [.08, .31, .54, .77]
    for i, ((head, body), x) in enumerate(zip(labels, xs)):
        ax.add_patch(FancyBboxPatch((x, y-.13), .16, .115, boxstyle="round,pad=.012", facecolor=LIGHT, edgecolor=CYAN, transform=ax.transAxes))
        ax.text(x+.08, y-.045, head, ha="center", fontsize=9, weight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(x+.08, y-.083, body, ha="center", va="top", fontsize=8, color=INK, linespacing=1.3, transform=ax.transAxes)
        if i < 3: ax.annotate("", xy=(xs[i+1]-.012, y-.072), xytext=(x+.172, y-.072), arrowprops={"arrowstyle": "->", "color": GREY}, xycoords=ax.transAxes)
    y -= .19
    para(ax, "Pour SARIMAX opérationnel, les valeurs futures d'ETP et SWI ne sont pas lues après 1999 : elles sont d'abord extrapolées par deux SARIMA entraînés sur 1970–1999, puis injectées dans le modèle de température.", .06, y, 103, 9.3)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def model_selection(pdf, page):
    fig, ax = base_page("Sélection des familles de modèles", "Pourquoi la saisonnalité et les variables exogènes améliorent la prévision", "03 — Modèles")
    y = section_label(ax, "Structures retenues", .83)
    rows = [
        ["ARIMA", "(1,0,1)", "—", "T passée", "Référence non saisonnière"],
        ["SARIMA", "(1,0,2)", "(0,1,1,12)", "T passée", "Modèle final univarié"],
        ["SARIMAX", "(0,0,0)", "(0,1,1,12)", "T + ETP + SWI", "Meilleur backtest mensuel"],
    ]
    draw_table(fig, [.07, .62, .86, .18], ["Famille", "Ordre", "Saisonnier", "Information", "Rôle"], rows, [.13,.15,.19,.20,.25], 8)
    y = .565
    y = section_label(ax, "Lecture statistique", y)
    y = bullet_list(ax, [
        "ARIMA ne modélise pas explicitement le cycle annuel ; son test exploratoire initial est nettement moins performant.",
        "SARIMA ajoute une différenciation et une composante moyenne mobile saisonnières à période 12, adaptées aux températures mensuelles.",
        "SARIMAX conserve cette structure saisonnière et ajoute ETP/SWI standardisés, ce qui réduit l'erreur et améliore l'interprétabilité physique.",
    ], .06, y, 100, 9.4, .018)
    ax.add_patch(FancyBboxPatch((.06, .18), .88, .15, boxstyle="round,pad=.014", facecolor=LIGHT, edgecolor="#D6E3EA", transform=ax.transAxes))
    ax.text(.085, .295, "PRINCIPE DE SÉLECTION", fontsize=8.5, weight="bold", color=CYAN, transform=ax.transAxes)
    para(ax, "La décision finale repose sur le RMSE et le MAE hors échantillon, le biais, la robustesse spatiale et la comparaison point par point. AIC/BIC servent au diagnostic d'ajustement, mais ne remplacent pas la validation temporelle.", .085, .26, 100, 9.2)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def sarima_results(pdf, page):
    fig, ax = base_page("Performance de SARIMA", "Backtest 1996–1999 sur l'ensemble des 9 892 points", "04 — Résultats")
    metric_card(ax, .06, .83, .2, .13, f"{SARIMA['rmse_mean']:.3f} °C", "RMSE moyen", f"Médiane {SARIMA['rmse_median']:.3f} °C", BLUE)
    metric_card(ax, .285, .83, .2, .13, f"{SARIMA['mae_mean']:.3f} °C", "MAE moyen", "Erreur absolue mensuelle", CYAN)
    metric_card(ax, .51, .83, .2, .13, f"{SARIMA['bias_mean']:+.3f} °C", "Biais moyen", "Sous-estimation légère", RED)
    metric_card(ax, .735, .83, .2, .13, f"{SARIMA['share_rmse_le_1_5']*100:.1f} %", "RMSE ≤ 1,5 °C", "Part des points", GOLD)
    add_image(fig, RESULTS / "predictions_2000_2025/report/01_backtest_sarima_vs_naive.png", [.07, .29, .86, .36])
    para(ax, f"Interprétation — SARIMA bat le naïf saisonnier sur {FORECAST_S['sarima_better_rmse_points_pct']:.2f} % des points. Le gain RMSE médian atteint {FORECAST_S['median_rmse_gain_vs_naive_pct']:.1f} %. La distribution reste concentrée : 90 % des points sont entre {FORECAST_S['sarima_rmse_p05']:.3f} et {FORECAST_S['sarima_rmse_p95']:.3f} °C.", .07, .245, 105, 9.2)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def sarima_space(pdf, page):
    fig, ax = base_page("Comportement temporel et spatial de SARIMA", "Cycle saisonnier, trajectoire moyenne et hétérogénéité géographique", "04 — Résultats")
    add_image(fig, RESULTS / "predictions_2000_2025/report/02_historique_et_previsions_1960_2025.png", [.06, .54, .88, .29])
    add_image(fig, RESULTS / "predictions_2000_2025/report/04_cartes_previsions.png", [.10, .17, .80, .31])
    para(ax, f"La moyenne projetée passe de {FORECAST_S['forecast_mean_2000_2004_c']:.2f} °C en 2000–2004 à {FORECAST_S['forecast_mean_2021_2025_c']:.2f} °C en 2021–2025, soit {FORECAST_S['forecast_change_last5_minus_first5_c']:+.3f} °C. Ce quasi-plateau est cohérent avec un modèle saisonnier sans scénario climatique externe : il prolonge les régularités historiques, pas une trajectoire climatique forcée.", .07, .12, 105, 8.8)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def sarimax_results(pdf, page):
    fig, ax = base_page("Performance de SARIMAX", "Comparaison équitable avec SARIMA sur les mêmes points et périodes", "05 — SARIMAX")
    metric_card(ax, .06, .83, .2, .13, f"{SARIMAX['rmse_mean']:.3f} °C", "RMSE moyen", f"Médiane {SARIMAX['rmse_median']:.3f} °C", RED)
    metric_card(ax, .285, .83, .2, .13, f"{SARIMAX['mae_mean']:.3f} °C", "MAE moyen", "Erreur absolue mensuelle", CYAN)
    metric_card(ax, .51, .83, .2, .13, f"{SARIMAX['rmse_improvement_vs_sarima']*100:.1f} %", "Gain RMSE", "Face à SARIMA", GOLD)
    metric_card(ax, .735, .83, .2, .13, f"{SARIMAX['sarimax_win_rate']*100:.1f} %", "Victoires", "Comparaison point par point", BLUE)
    add_image(fig, RESULTS / "sarimax_all_9892_compact/figures/05_comparaison_SARIMA_SARIMAX.png", [.10, .39, .80, .25])
    add_image(fig, RESULTS / "sarimax_all_9892_compact/figures/07_gains_SARIMAX_point_par_point.png", [.10, .10, .80, .25])
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def exogenous(pdf, page):
    fig, ax = base_page("Apport d’ETP et de SWI", "Interprétation des coefficients exogènes standardisés", "05 — SARIMAX")
    metric_card(ax, .08, .82, .24, .14, f"{SARIMAX['coef_ETP_mean']:.3f}", "Coefficient ETP moyen", "Positif sur 100 % des points", RED)
    metric_card(ax, .38, .82, .24, .14, f"{SARIMAX['coef_SWI_mean']:.3f}", "Coefficient SWI moyen", f"Positif sur {SARIMAX['coef_SWI_positive_share']*100:.1f} %", CYAN)
    metric_card(ax, .68, .82, .24, .14, "ETP", "Variable dominante", "Après standardisation", GOLD)
    add_image(fig, RESULTS / "sarimax_all_9892_compact/figures/03_cartes_coefficients_ETP_SWI.png", [.08, .32, .84, .33])
    y = .265
    y = para(ax, "L'effet moyen positif d'ETP est cohérent avec la saison chaude et le bilan énergétique. SWI présente un effet plus faible et plus hétérogène ; son coefficient ne doit pas être interprété comme une causalité isolée, car ETP, SWI et saisonnalité partagent une structure temporelle commune.", .07, y, 105, 9.3)
    y -= .018
    para(ax, "La standardisation est essentielle : les coefficients comparent des variations d'un écart-type, et non les unités brutes d'ETP et SWI.", .07, y, 105, 9.1, GREY)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def forecast_comparison(pdf, page, monthly_s, monthly_x):
    fig, ax0 = base_page("Prévisions 2000–2025 : SARIMA vs SARIMAX", "Comparaison des moyennes spatiales et de l'incertitude ponctuelle", "06 — Prévisions")
    ax = fig.add_axes([.09, .52, .82, .28])
    ax.plot(monthly_s.DT, monthly_s["T"], color=BLUE, lw=1.6, label="SARIMA")
    ax.plot(monthly_x.DT, monthly_x["T"], color=RED, lw=1.4, label="SARIMAX")
    ax.set_ylabel("Température moyenne (°C)"); ax.grid(alpha=.2); ax.legend(ncol=2); ax.set_title("Moyenne mensuelle sur 9 892 points", fontsize=10, weight="bold")
    ax2 = fig.add_axes([.09, .20, .82, .23])
    ax2.plot(monthly_s.DT, monthly_s.WIDTH, color=BLUE, label="SARIMA")
    ax2.plot(monthly_x.DT, monthly_x.WIDTH, color=RED, label="SARIMAX")
    ax2.set_ylabel("Largeur moyenne IC 95 % (°C)"); ax2.set_xlabel("Année"); ax2.grid(alpha=.2); ax2.legend(ncol=2); ax2.set_title("Évolution de l'incertitude prédictive ponctuelle", fontsize=10, weight="bold")
    mean_s, mean_x = monthly_s["T"].mean(), monthly_x["T"].mean()
    para(ax0, f"Moyenne 2000–2025 : SARIMA {mean_s:.3f} °C ; SARIMAX {mean_x:.3f} °C ; écart moyen {mean_x-mean_s:+.3f} °C. Les trajectoires proches indiquent que l'extrapolation d'ETP/SWI conserve principalement la saisonnalité historique.", .07, .145, 105, 9.1)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def operational(pdf, page):
    fig, ax = base_page("Production, stockage et contrôles", "Des sorties compactes, vérifiées et reproductibles", "07 — Industrialisation")
    rows = [
        ["SARIMA T", "3 086 304", "79,41 Mo", "0", "9 862 / 9 892"],
        ["SARIMAX T", "3 086 304", "63,47 Mo", "0", "9 892 / 9 892"],
        ["SARIMAX ETP/SWI", "3 086 304", "19,34 Mo", "0", "ETP 9 891 ; SWI 9 620"],
        ["Paramètres SARIMA", "9 892", "2,32 Mo", "0", "CSV compact"],
        ["Paramètres SARIMAX", "9 892", "4,81 Mo", "0", "CSV compact"],
    ]
    draw_table(fig, [.07, .55, .86, .25], ["Sortie", "Lignes", "Taille", "Manquants", "Convergence / format"], rows, [.21,.17,.13,.14,.27], 7.6)
    y = section_label(ax, "Contrôles réalisés", .50)
    y = bullet_list(ax, [
        "Comptage exact : 9 892 points × 312 mois = 3 086 304 lignes par fichier de température.",
        "Dates contrôlées de 200001 à 202512 ; aucune cellule manquante et aucun intervalle inversé.",
        "Sauvegarde des coefficients et échelles plutôt que des objets Statsmodels complets, afin de réduire fortement le volume.",
        "Scripts relançables et reprise automatique des points déjà terminés.",
        "Versionnement GitHub des scripts, métriques, paramètres, figures et archives de prévisions compatibles avec la limite de taille.",
    ], .06, y, 101, 9.3, .018)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def limitations(pdf, page):
    fig, ax = base_page("Limites scientifiques et interprétation", "Ce que les résultats permettent — et ne permettent pas — d'affirmer", "08 — Discussion")
    y = section_label(ax, "Conclusions défendables", .83)
    y = bullet_list(ax, [
        "SARIMA est une référence mensuelle robuste et surpasse très nettement le naïf saisonnier.",
        "SARIMAX ETP+SWI est le meilleur modèle statistique mensuel évalué sur les 9 892 points dans ce protocole.",
        "Le gain de SARIMAX est spatialement généralisé et ses coefficients fournissent une lecture interprétable.",
    ], .06, y, 101, 9.5, .018)
    y -= .02
    y = section_label(ax, "Limites à annoncer explicitement", y)
    y = bullet_list(ax, [
        "Les observations s'arrêtent en 1999 : les sorties 2000–2025 sont des projections non vérifiées contre la vérité observée locale.",
        "SARIMAX dépend d'ETP et SWI futurs. Dans la chaîne opérationnelle, ces variables sont elles-mêmes prévues, ce qui propage leur incertitude.",
        "272 modèles SWI présentent un avertissement de convergence ; les températures SARIMAX finales convergent néanmoins aux 9 892 points.",
        "Les intervalles produits par SARIMAX conditionnent principalement l'incertitude de T et ne propagent pas complètement l'incertitude des trajectoires ETP/SWI.",
        "Aucune supériorité générale sur TCN, ConvLSTM, XGBoost, PatchTST ou les modèles climatiques ne peut être revendiquée sans benchmark commun.",
    ], .06, y, 101, 9.3, .016)
    ax.add_patch(FancyBboxPatch((.06, .10), .88, .12, boxstyle="round,pad=.015", facecolor="#FDECEC", edgecolor="#EAB8B8", transform=ax.transAxes))
    ax.text(.085, .19, "FORMULATION RECOMMANDÉE", fontsize=8.5, color=RED, weight="bold", transform=ax.transAxes)
    para(ax, "« SARIMAX est le meilleur modèle statistique mensuel évalué dans notre protocole sur 9 892 points, avec un RMSE moyen de 1,278 °C et un gain de 11,5 % face à SARIMA. »", .085, .16, 98, 9.2, INK, "bold")
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def recommendations(pdf, page):
    fig, ax = base_page("Recommandations et feuille de route", "Étapes prioritaires pour transformer l'étude en système de prévision validé", "09 — Recommandations")
    items = [
        ("1", "Validation récente", "Importer les observations T, ETP et SWI 2000–2025 et calculer RMSE, MAE, biais, couverture des IC et performance sur extrêmes."),
        ("2", "Backtest opérationnel SARIMAX", "Prévoir ETP/SWI dans chaque fenêtre de validation, afin de mesurer l'erreur réellement propagée jusqu'à T."),
        ("3", "Benchmark commun", "Comparer SARIMA, SARIMAX, XGBoost, TCN, ConvLSTM et PatchTST sur la même fréquence mensuelle, les mêmes points, dates et horizons."),
        ("4", "Incertitude complète", "Utiliser scénarios ou simulations Monte-Carlo d'ETP/SWI pour obtenir des intervalles SARIMAX intégrant l'incertitude exogène."),
        ("5", "Suivi en production", "Versionner données, paramètres, métriques et dérive ; automatiser les contrôles et la régénération des figures."),
    ]
    y = .82
    for num, head, body in items:
        ax.add_patch(plt.Circle((.09, y-.025), .025, color=CYAN, transform=ax.transAxes))
        ax.text(.09, y-.025, num, ha="center", va="center", color=WHITE, fontsize=10, weight="bold", transform=ax.transAxes)
        ax.text(.135, y, head, color=BLUE, fontsize=11, weight="bold", va="top", transform=ax.transAxes)
        para(ax, body, .135, y-.035, 88, 9.1)
        y -= .135
    ax.add_patch(FancyBboxPatch((.06, .08), .88, .10, boxstyle="round,pad=.015", facecolor=BLUE, edgecolor=BLUE, transform=ax.transAxes))
    ax.text(.085, .145, "PRIORITÉ IMMÉDIATE", color="#A9DCE9", fontsize=8.5, weight="bold", transform=ax.transAxes)
    ax.text(.085, .112, "Récupérer les observations 2000–2025 pour convertir les projections en évaluation vérifiable.", color=WHITE, fontsize=9.3, weight="bold", transform=ax.transAxes)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def appendix(pdf, page):
    fig, ax = base_page("Annexe — livrables et reproductibilité", "Emplacements principaux dans le dépôt climate_project", "10 — Annexe")
    y = .82
    groups = [
        ("Scripts HERGI", [
            "archive_old/sarima_sarimax/scripts/arima_sarima/predict_sarima_2000_2025.py",
            "archive_old/sarima_sarimax/scripts/arima_sarima/generate_prediction_2000_2025_report.py",
            "archive_old/sarima_sarimax/scripts/sarimax/predict_sarimax_2000_2025.py",
        ]),
        ("Prévisions SARIMA", [
            "archive_old/sarima_sarimax_results/predictions_2000_2025/temperature_predictions_2000_2025.csv.gz",
            "archive_old/sarima_sarimax_results/predictions_2000_2025/sarima_parameters_1960_1999.csv",
        ]),
        ("Prévisions SARIMAX", [
            "archive_old/sarima_sarimax_results/sarimax_predictions_2000_2025/temperature_predictions_2000_2025.csv.gz",
            "archive_old/sarima_sarimax_results/sarimax_predictions_2000_2025/etp_swi_predictions_2000_2025.csv.gz",
            "archive_old/sarima_sarimax_results/sarimax_predictions_2000_2025/sarimax_parameters_1970_1999.csv",
        ]),
        ("Rapports et contrôles", [
            "archive_old/sarima_sarimax_results/predictions_2000_2025/report/",
            "archive_old/sarima_sarimax_results/*/evaluation_summary.json",
            "archive_old/sarima_sarimax_results/*/validation_summary.json",
        ]),
    ]
    for head, paths in groups:
        ax.text(.06, y, head, color=CYAN, fontsize=9.5, weight="bold", va="top", transform=ax.transAxes); y -= .035
        for path in paths:
            ax.add_patch(FancyBboxPatch((.07, y-.032), .85, .038, boxstyle="round,pad=.006", facecolor=LIGHT, edgecolor="#D6E3EA", transform=ax.transAxes))
            ax.text(.085, y-.012, path, color=INK, fontsize=7.8, family="monospace", va="center", transform=ax.transAxes); y -= .052
        y -= .025
    para(ax, "Dépôt public : https://github.com/dianguehergi/climate_project", .06, .14, 100, 9.5, BLUE, "bold")
    para(ax, "Le présent rapport a été généré automatiquement à partir des métriques et figures versionnées, afin d'assurer la traçabilité entre chiffres publiés et artefacts de calcul.", .06, .10, 104, 8.6, GREY)
    footer(fig, page); pdf.savefig(fig); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Agrégation des prévisions SARIMA...", flush=True)
    monthly_s = monthly_aggregate(RESULTS / "predictions_2000_2025/temperature_predictions_2000_2025.csv.gz")
    print("Agrégation des prévisions SARIMAX...", flush=True)
    monthly_x = monthly_aggregate(RESULTS / "sarimax_predictions_2000_2025/temperature_predictions_2000_2025.csv.gz")
    with PdfPages(PDF, metadata={
        "Title": "Rapport final HERGI — ARIMA, SARIMA et SARIMAX",
        "Author": "Projet Climate — HERGI",
        "Subject": "Modélisation mensuelle de la température sur 9 892 points",
        "Keywords": "ARIMA SARIMA SARIMAX température prévision SAFRAN",
    }) as pdf:
        pages = [cover, executive, data_method, model_selection, sarima_results, sarima_space, sarimax_results, exogenous]
        page = 1
        for fn in pages:
            fn(pdf, page); page += 1
        forecast_comparison(pdf, page, monthly_s, monthly_x); page += 1
        operational(pdf, page); page += 1
        limitations(pdf, page); page += 1
        recommendations(pdf, page); page += 1
        appendix(pdf, page)
    print(f"PDF créé : {PDF}")


if __name__ == "__main__":
    main()
