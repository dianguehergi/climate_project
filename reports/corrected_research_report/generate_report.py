#!/usr/bin/env python3
"""Génère le rapport scientifique corrigé en PDF A4 (exactement 11 pages)."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF = OUT / "Rapport_Projet_Recherche_FR_corrige_11_pages.pdf"

NAVY = "#12324A"
BLUE = "#1976A3"
CYAN = "#48A9C5"
GREEN = "#2A9D78"
ORANGE = "#E28C3C"
RED = "#C84B4B"
INK = "#1E2B33"
MUTED = "#5D6A72"
PALE = "#EEF5F7"
WHITE = "#FFFFFF"

IMG = {
    "sarima_dist": ROOT / "archive_old/sarima_sarimax_results/all_9892_best_model/figures/01_distributions_erreurs.png",
    "sarima_maps": ROOT / "archive_old/sarima_sarimax_results/all_9892_best_model/figures/02_cartes_spatiales.png",
    "sarimax_compare": ROOT / "archive_old/sarima_sarimax_results/sarimax_all_9892_compact/figures/05_comparaison_SARIMA_SARIMAX.png",
    "sarimax_gain": ROOT / "archive_old/sarima_sarimax_results/sarimax_all_9892_compact/figures/07_gains_SARIMAX_point_par_point.png",
    "sarimax_coef": ROOT / "archive_old/sarima_sarimax_results/sarimax_all_9892_compact/figures/03_cartes_coefficients_ETP_SWI.png",
    "forecast": ROOT / "archive_old/sarima_sarimax_results/predictions_2000_2025/report/02_historique_et_previsions_1960_2025.png",
    "convlstm": ROOT / "archive_old/tcn_convlstm/outputs/figures/convlstm_spatial_center_predictions.png",
    "convlstm_map": ROOT / "archive_old/tcn_convlstm/outputs/figures/map_9892_convlstm_mae.png",
    "xgb": ROOT / "results/model_comparison/figures/Figure_01_global_error_comparison.png",
}


def page(title, section, number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0, .965), 1, .035, color=NAVY))
    ax.text(.06, .943, section.upper(), fontsize=7.8, color=BLUE, weight="bold")
    ax.text(.06, .902, title, fontsize=18, color=NAVY, weight="bold", va="top")
    ax.plot([.06, .94], [.872, .872], color=CYAN, lw=1.1)
    ax.text(.06, .025, "Projet Climate — SAFRAN France | Rapport corrigé", fontsize=7.2, color=MUTED)
    ax.text(.94, .025, f"{number} / 11", fontsize=7.2, color=MUTED, ha="right")
    return fig, ax


def para(ax, text, x, y, width=92, size=8.7, color=INK, weight="normal", spacing=1.25):
    wrapped = textwrap.fill(text, width=width, break_long_words=False)
    ax.text(x, y, wrapped, fontsize=size, color=color, weight=weight, va="top", linespacing=spacing)
    lines = wrapped.count("\n") + 1
    return y - lines * size / 720 * spacing - .012


def heading(ax, text, x, y, size=11.5):
    ax.text(x, y, text, fontsize=size, color=BLUE, weight="bold", va="top")
    return y - .032


def card(ax, x, y, w, h, title, value, note="", color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=PALE, edgecolor="#D4E4E9", lw=.8))
    ax.add_patch(Rectangle((x, y), .008, h, color=color))
    ax.text(x+.022, y+h-.025, title, fontsize=7.5, color=MUTED, weight="bold", va="top")
    ax.text(x+.022, y+h-.062, value, fontsize=17, color=color, weight="bold", va="top")
    if note:
        ax.text(x+.022, y+.018, textwrap.fill(note, 30), fontsize=6.8, color=INK, va="bottom")


def image(ax, path, box, caption=None):
    x, y, w, h = box
    if path.exists():
        arr = plt.imread(path)
        iax = ax.figure.add_axes([x, y, w, h])
        iax.imshow(arr); iax.axis("off")
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F5F5F5", edgecolor="#CCCCCC"))
        ax.text(x+w/2, y+h/2, "Figure indisponible", ha="center", color=MUTED, fontsize=8)
    if caption:
        ax.text(x, y-.012, textwrap.fill(caption, 110), fontsize=6.7, color=MUTED, va="top")


def table(ax, rows, cols, box, widths=None, size=7.2, header_color=NAVY):
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", colLoc="center", bbox=box,
                 colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(size)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#CBD9DE"); cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor(header_color); cell.get_text().set_color(WHITE); cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F3F7F8")
    return t


def save(fig, pdf):
    pdf.savefig(fig, dpi=220, facecolor=WHITE)
    plt.close(fig)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF, metadata={
        "Title": "Local Climate Downscaling over France — Rapport corrigé",
        "Author": "Hergi DIANGUE, Yann DE-NZOGHE, Owen MOUKETOU, Ousmane MOMBO MOMBO",
        "Subject": "SARIMA/SARIMAX, SAFRAN et diagnostic de correction de biais",
    }) as pdf:
        # 1 — Couverture
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=WHITE)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.add_patch(Rectangle((0, .68), 1, .32, color=NAVY))
        ax.add_patch(Rectangle((.06, .63), .20, .018, color=CYAN))
        ax.text(.06, .90, "RESEARCH PROJECT REPORT", fontsize=11, color=CYAN, weight="bold")
        ax.text(.06, .82, "Local Climate Downscaling\nover France", fontsize=30, color=WHITE, weight="bold", va="top")
        ax.text(.06, .70, "Pipeline SARIMA/SARIMAX sur données SAFRAN\net diagnostic transparent de correction de biais", fontsize=13, color="#DCEBF0", va="top")
        ax.text(.06, .55, "Version scientifique corrigée — 11 pages", fontsize=13, color=BLUE, weight="bold")
        ax.text(.06, .47, "Auteurs", fontsize=9, color=MUTED, weight="bold")
        ax.text(.06, .44, "Hergi DIANGUE  ·  Yann DE-NZOGHE\nOwen MOUKETOU  ·  Ousmane MOMBO MOMBO", fontsize=11, color=INK, linespacing=1.5)
        ax.text(.06, .34, "Encadrement", fontsize=9, color=MUTED, weight="bold")
        ax.text(.06, .31, "Émile EGRETEAU-DRUET\nAI Clinic (Clinique de l’IA) — Responsable : Anuradha Kar", fontsize=10, color=INK, linespacing=1.5)
        ax.text(.06, .16, "31 août 2026", fontsize=10, color=MUTED)
        ax.text(.94, .04, "1 / 11", fontsize=7.2, color=MUTED, ha="right")
        save(fig, pdf)

        # 2 — Sommaire et résumé
        fig, ax = page("Sommaire et résumé exécutif", "Vue d’ensemble", 2)
        rows = [
            ["1", "Introduction et question scientifique", "3"],
            ["2", "État de l’art et positionnement", "4"],
            ["3", "Données, protocole et métriques", "5–6"],
            ["4", "Résultats SARIMA et SARIMAX", "7–8"],
            ["5", "Projections et correction de biais", "9"],
            ["6", "ML/DL, limites et perspectives", "10"],
            ["7", "Conclusion et références", "11"],
        ]
        table(ax, rows, ["Section", "Contenu", "Page"], [.08,.56,.84,.27], widths=[.15,.68,.17], size=8)
        y = heading(ax, "Résumé exécutif", .08, .51)
        y = para(ax, "Ce projet construit un pipeline reproductible de modélisation mensuelle de la température sur les 9 892 points de la grille SAFRAN. Le backtest strict 1996–1999 montre que SARIMA constitue une référence robuste (RMSE 1,445 °C) et que SARIMAX avec ETP et SWI améliore encore la précision (RMSE 1,278 °C, gain 11,5 %).", .08, y, 105, 9.2)
        y = para(ax, "Les projections 2000–2025 prolongent la dynamique historique, mais ne constituent pas une projection forcée du changement climatique. Le post-traitement par quantile mapping réalisé dans les travaux utilise les observations 2000–2025 pour établir les quantiles de référence : ses métriques sont donc conservées comme diagnostic rétrospectif, et non comme validation indépendante. Cette correction explicite est essentielle à l’intégrité scientifique du rapport.", .08, y, 105, 9.2)
        card(ax,.08,.16,.25,.13,"COUVERTURE","9 892","points SAFRAN",BLUE)
        card(ax,.375,.16,.25,.13,"MEILLEUR BACKTEST","1,278 °C","RMSE SARIMAX",GREEN)
        card(ax,.67,.16,.25,.13,"HORIZON","312 mois","projections 2000–2025",ORANGE)
        save(fig, pdf)

        # 3 — Introduction
        fig, ax = page("Introduction et problématique", "1 — Introduction", 3)
        y = heading(ax, "1.1 Contexte", .07, .84)
        y = para(ax, "L’adaptation au changement climatique nécessite une information locale, cohérente et quantifiée. Les modèles globaux et régionaux fonctionnent à des résolutions plus grossières que celles attendues pour de nombreuses décisions territoriales. La descente d’échelle vise à relier cette information de grande échelle aux caractéristiques climatiques locales.", .07, y, 106, 9)
        y = para(ax, "Le projet utilise la réanalyse SAFRAN de Météo-France comme référence spatiale sur la France métropolitaine : 9 892 points, espacés d’environ 8 km. La cible est la température à 2 mètres au pas mensuel. Les données 1960–1999 servent à la construction des modèles ; les observations SIM/SAFRAN 2000–2025 permettent une confrontation rétrospective des projections.", .07, y, 106, 9)
        y = heading(ax, "1.2 Question scientifique", .07, y-.005)
        ax.add_patch(FancyBboxPatch((.07,y-.105),.86,.095,boxstyle="round,pad=.012",facecolor=PALE,edgecolor=CYAN))
        para(ax, "Dans quelle mesure des modèles statistiques saisonniers peuvent-ils reconstruire la température mensuelle locale sur toute la grille SAFRAN, et quelles précautions sont nécessaires pour corriger leurs biais sans contaminer l’évaluation ?", .095, y-.03, 92, 10.5, NAVY, "bold")
        y -= .145
        y = heading(ax, "1.3 Objectifs", .07, y)
        bullets = [
            "Construire un pipeline SARIMA/SARIMAX reproductible et parallélisable sur 9 892 points.",
            "Comparer les modèles sur un backtest temporel strict et une référence saisonnière naïve.",
            "Produire 312 mois de projections et caractériser leurs limites climatiques.",
            "Évaluer les pistes XGBoost et ConvLSTM sans mélanger prévision J+1 et projection climatique.",
            "Distinguer un diagnostic de recalage a posteriori d’une véritable validation indépendante.",
        ]
        for b in bullets:
            ax.text(.09,y,"•",fontsize=12,color=BLUE,va="top"); y=para(ax,b,.12,y,90,8.6)
        save(fig, pdf)

        # 4 — État de l'art
        fig, ax = page("État de l’art et positionnement", "1.2 — Littérature", 4)
        y = para(ax, "La littérature récente distingue la descente d’échelle statistique, l’émulation de modèles régionaux et la correction de biais. Elle souligne surtout la généralisation hors distribution : un modèle entraîné sur une climatologie historique ne peut pas inventer un forçage futur absent de ses entrées.", .07, .84, 106, 9)
        rows = [
            ["Rampal et al. (2024)", "Revue ML/DL", "Généralisation hors distribution"],
            ["Hernanz et al. (2023)", "ML vs RCM", "Comparaison multi-variable Europe"],
            ["Ozbuldu et al. (2025)", "RF, SVR / CMIP6", "RMSE T ≈ 1,44–2,23 °C"],
            ["Loganathan et al. (2025)", "GeoSTANet", "RMSE rapportée 1,57 °C"],
        ]
        table(ax, rows, ["Étude", "Approche", "Apport au cadrage"], [.07,.55,.86,.22], widths=[.28,.25,.47], size=7.4)
        y = heading(ax, "Lecture correcte des comparaisons", .07, .50)
        y = para(ax, "Les valeurs publiées donnent des ordres de grandeur utiles, mais ne forment pas un classement direct : régions, périodes, variables, résolutions et protocoles diffèrent. Le RMSE de 1,278 °C obtenu ici positionne favorablement SARIMAX dans son propre protocole ; il ne démontre pas une supériorité générale sur GeoSTANet, SVR ou les RCM.", .07, y, 106, 9.1)
        y = heading(ax, "Conséquence méthodologique", .07, y-.005)
        y = para(ax, "La performance principale doit provenir du backtest 1996–1999. Toute correction de biais évaluée sur 2000–2025 doit être calibrée sur une période indépendante. Faute de cette séparation dans l’implémentation actuelle, les résultats corrigés 2000–2025 sont rapportés uniquement comme diagnostic post-hoc.", .07, y, 106, 9.1, RED, "bold")
        card(ax,.07,.15,.27,.13,"PRINCIPE","Pas de fuite","calibration ≠ évaluation",RED)
        card(ax,.365,.15,.27,.13,"COMPARAISON","Même protocole","avant tout classement",BLUE)
        card(ax,.66,.15,.27,.13,"CIBLE","Climatologie","pas météo J+1",GREEN)
        save(fig, pdf)

        # 5 — Données et protocole
        fig, ax = page("Données et protocole expérimental", "2 — Méthodologie", 5)
        card(ax,.07,.72,.25,.12,"DONNÉES JOURNALIÈRES","144 522 120","lignes 1960–1999",BLUE)
        card(ax,.375,.72,.25,.12,"SÉRIE MENSUELLE","4 748 160","9 892 × 480 mois",GREEN)
        card(ax,.68,.72,.25,.12,"VARIABLES","T, ETP, SWI","cible + exogènes",ORANGE)
        y = heading(ax, "2.1 Préparation", .07, .66)
        y = para(ax, "La température journalière est agrégée en moyenne mensuelle. Les contrôles portent sur la continuité temporelle, les doublons, les valeurs manquantes et l’unicité des couples de coordonnées SAFRAN. ETP et SWI sont retenus comme variables exogènes pour leur information statistique et leur cohérence physique avec le bilan d’énergie et le couplage sol–atmosphère.", .07, y, 106, 8.7)
        y = heading(ax, "2.2 Découpage temporel réellement utilisé", .07, y-.002)
        rows = [
            ["Comparaison équitable", "1970–1995", "1996–1999", "SARIMA vs SARIMAX"],
            ["SARIMA historique complet", "1960–1995", "1996–1999", "Référence vs naïf"],
            ["Ajustement final SARIMA", "1960–1999", "—", "Projection 2000–2025"],
            ["Ajustement final SARIMAX", "1970–1999", "—", "ETP/SWI futurs extrapolés"],
        ]
        table(ax, rows, ["Expérience", "Apprentissage", "Test", "Usage"], [.07,.36,.86,.22], widths=[.27,.20,.18,.35], size=7.2)
        y = heading(ax, "2.3 Prévention des fuites", .07, .31)
        para(ax, "Le backtest statistique est hors échantillon. En revanche, le quantile mapping post-hoc disponible dans le dépôt calcule les quantiles observés sur 2000–2025, la même période que celle évaluée. Ses métriques ne sont donc pas utilisées comme preuve principale de généralisation.", .07, y, 106, 9, RED, "bold")
        save(fig, pdf)

        # 6 — Modèles et métriques
        fig, ax = page("Modèles, mise en œuvre et métriques", "2 — Méthodologie", 6)
        rows = [
            ["ARIMA", "(1,0,1)", "Aucune", "Référence non saisonnière"],
            ["SARIMA", "(1,0,2)(0,1,1,12)", "Aucune", "Cycle annuel explicite"],
            ["SARIMAX", "(0,0,0)(0,1,1,12)", "ETP, SWI", "Exogènes standardisées"],
        ]
        table(ax, rows, ["Modèle", "Ordres", "Entrées", "Rôle"], [.07,.67,.86,.15], widths=[.18,.28,.20,.34], size=7.5)
        y = heading(ax, "2.3 Mise en œuvre", .07, .62)
        y = para(ax, "Un modèle est ajusté par point. Les traitements sont parallèles et reprenables ; les paramètres, métriques et erreurs sont sauvegardés sous forme compacte. Pour les projections SARIMAX 2000–2025, ETP et SWI ne sont pas connus : deux SARIMA auxiliaires entraînés sur 1970–1999 fournissent les exogènes futurs, ce qui introduit une incertitude supplémentaire non entièrement propagée.", .07, y, 106, 8.8)
        y = heading(ax, "2.4 Métriques", .07, y)
        rows = [
            ["MAE", "Erreur absolue moyenne", "Précision typique en °C"],
            ["RMSE", "Racine de l’erreur quadratique", "Pénalise les grandes erreurs"],
            ["Biais", "Moyenne (prévision − observation)", "Erreur systématique"],
            ["R²", "Part de variance expliquée", "Qualité globale de restitution"],
        ]
        table(ax, rows, ["Indicateur", "Définition", "Interprétation"], [.07,.31,.86,.21], widths=[.18,.42,.40], size=7.4)
        y = heading(ax, "Diagnostic statistique", .07, .26)
        para(ax, "Les résidus du point illustratif sont contrôlés avec Ljung–Box, Jarque–Bera et un test d’hétéroscédasticité. Ces diagnostics complètent la validation temporelle ; ils ne la remplacent jamais.", .07, y, 106, 8.8)
        save(fig, pdf)

        # 7 — SARIMA
        fig, ax = page("Résultats SARIMA sur la grille complète", "3 — Résultats", 7)
        card(ax,.07,.72,.25,.12,"RMSE MOYEN","1,445 °C","médiane 1,441 °C",BLUE)
        card(ax,.375,.72,.25,.12,"MAE MOYEN","1,166 °C","biais −0,198 °C",GREEN)
        card(ax,.68,.72,.25,.12,"GAIN VS NAÏF","30,4 %","gain RMSE médian",ORANGE)
        image(ax, IMG["sarima_dist"], [.075,.40,.85,.27], "Figure 1 — Distribution des erreurs SARIMA sur les 9 892 points, backtest 1996–1999.")
        image(ax, IMG["sarima_maps"], [.075,.10,.85,.24], "Figure 2 — Structure spatiale des performances SARIMA sur la grille SAFRAN.")
        ax.text(.07,.365,"SARIMA bat la référence saisonnière naïve sur 99,99 % des points ; 90 % des RMSE se situent entre 1,188 et 1,679 °C.",fontsize=8.2,color=INK,weight="bold")
        save(fig, pdf)

        # 8 — SARIMAX
        fig, ax = page("SARIMAX : apport d’ETP et SWI", "3 — Résultats", 8)
        card(ax,.07,.72,.25,.12,"RMSE MOYEN","1,278 °C","−11,5 % vs SARIMA",GREEN)
        card(ax,.375,.72,.25,.12,"MAE MOYEN","1,016 °C","biais −0,133 °C",BLUE)
        card(ax,.68,.72,.25,.12,"TAUX DE VICTOIRE","95,8 %","points gagnés",ORANGE)
        image(ax, IMG["sarimax_compare"], [.075,.43,.85,.23], "Figure 3 — Comparaison équitable SARIMA/SARIMAX : mêmes points et même test 1996–1999.")
        image(ax, IMG["sarimax_gain"], [.075,.16,.85,.21], "Figure 4 — Gains SARIMAX point par point sur la grille complète.")
        para(ax, "Le coefficient ETP est positif sur 100 % des points ; SWI est positif sur 88,5 %. Ces signes sont cohérents avec le rôle énergétique d’ETP et le couplage sol–atmosphère, sans suffire à établir une causalité.", .075, .12, 106, 8.2)
        save(fig, pdf)

        # 9 — Projection et diagnostic post-hoc
        fig, ax = page("Projections 2000–2025 et correction de biais", "3 — Résultats", 9)
        image(ax, IMG["forecast"], [.07,.58,.86,.23], "Figure 5 — Historique et projections SARIMA 2000–2025. La dynamique historique est prolongée sans forçage climatique externe.")
        y = heading(ax, "Résultats bruts face aux observations 2000–2025", .07, .53)
        rows = [
            ["SARIMA brut", "1,525", "1,922", "−0,831", "0,909"],
            ["SARIMAX brut", "1,484", "1,848", "−0,789", "0,916"],
        ]
        table(ax, rows, ["Modèle", "MAE °C", "RMSE °C", "Biais °C", "R²"], [.07,.39,.86,.11], widths=[.30,.175,.175,.175,.175], size=7.5)
        y = heading(ax, "Diagnostic de quantile mapping — statut corrigé", .07, .35)
        y = para(ax, "Les scripts post-hoc calculent les quantiles de T_REAL sur 2000–2025 puis évaluent la correction sur ces mêmes données, avec un regroupement par mois à l’échelle globale. Les valeurs obtenues (SARIMAX : MAE 1,372 °C, RMSE 1,716 °C, biais −0,003 °C) décrivent la capacité de recalage in-sample ; elles ne constituent pas une validation indépendante et ne doivent pas être présentées comme la performance finale d’ADAMONT.", .07, y, 106, 8.7, RED, "bold")
        ax.add_patch(FancyBboxPatch((.07,.105),.86,.10,boxstyle="round,pad=.012",facecolor="#FFF3E8",edgecolor=ORANGE))
        para(ax, "Protocole requis pour la suite : calibrer une fonction par point et par mois sur une période historique commune, figer cette fonction, puis l’appliquer à une période indépendante. L’ADAMONT complet nécessiterait en plus des régimes de temps issus de champs atmosphériques de grande échelle.", .095, .18, 96, 8.7, INK, "bold")
        save(fig, pdf)

        # 10 — ML/DL et limites
        fig, ax = page("Apprentissage automatique, profond et limites", "4 — Discussion", 10)
        rows = [
            ["XGBoost baseline", "Mensuel, one-step", "1,423", "0,920", "Lags observés"],
            ["XGBoost avancé", "Mensuel, one-step", "1,377", "0,927", "Pas récursif"],
            ["ConvLSTM", "Quotidien J+1", "1,423", "0,930", "Bonne tâche court terme"],
            ["ConvLSTM récursif", "26 ans boucle fermée", "10,596", "−2,171", "Dérive sévère"],
        ]
        table(ax, rows, ["Modèle", "Protocole", "MAE °C", "R²", "Lecture"], [.06,.68,.88,.15], widths=[.22,.25,.14,.13,.26], size=6.9)
        image(ax, IMG["xgb"], [.075,.42,.40,.20], "Figure 6 — Résultats XGBoost : comparaison baseline/avancé.")
        image(ax, IMG["convlstm_map"], [.53,.42,.39,.20], "Figure 7 — Distribution spatiale de l’erreur ConvLSTM J+1.")
        y = heading(ax, "Limites principales", .07, .37)
        bullets = [
            "Absence de forçages climatiques futurs : les projections prolongent la climatologie historique.",
            "Exogènes SARIMAX elles-mêmes prévues, sans propagation complète de leur incertitude.",
            "Correction post-hoc actuelle contaminée par les observations de la période évaluée.",
            "Protocoles mensuels, J+1 et récursifs non directement comparables.",
            "Validation dédiée aux extrêmes climatiques encore à réaliser.",
        ]
        for b in bullets:
            ax.text(.08,y,"•",fontsize=11,color=RED,va="top"); y=para(ax,b,.11,y,94,8.2)
        save(fig, pdf)

        # 11 — Conclusion et références
        fig, ax = page("Conclusion, feuille de route et références", "4 — Conclusions", 11)
        y = heading(ax, "4.1 Conclusion", .07, .84)
        y = para(ax, "Le résultat central et solidement validé du projet est la supériorité de SARIMAX ETP+SWI sur SARIMA dans un backtest strict couvrant les 9 892 points : RMSE 1,278 °C contre 1,445 °C, soit un gain de 11,5 %. SARIMA reste une référence saisonnière robuste, légère et interprétable.", .07, y, 106, 8.8)
        y = para(ax, "La confrontation 2000–2025 révèle un biais froid des projections brutes. Le quantile mapping disponible montre qu’un recalage distributionnel peut réduire ce biais, mais son évaluation actuelle est in-sample. La prochaine version devra séparer calibration et validation avant de revendiquer une performance ADAMONT indépendante.", .07, y, 106, 8.8)
        y = heading(ax, "4.2 Feuille de route prioritaire", .07, y)
        rows = [
            ["1", "Refaire le quantile mapping avec calibration/validation séparées", "Critique"],
            ["2", "Ajouter ERA5 / EURO-CORDEX / CMIP et des régimes de temps", "Haute"],
            ["3", "Comparer tous les modèles sur cible, période et résolution identiques", "Haute"],
            ["4", "Évaluer canicules, vagues de froid et incertitudes", "Moyenne"],
        ]
        table(ax, rows, ["Priorité", "Action", "Niveau"], [.07,.37,.86,.18], widths=[.15,.67,.18], size=7.2)
        y = heading(ax, "Références", .07, .32)
        refs = [
            "Hernanz, A. et al. (2023). Comparison of machine learning statistical downscaling and regional climate models... International Journal of Climatology.",
            "Loganathan, P. et al. (2025). Regional climate projections using a deep-learning-based multi-model evaluation and downscaling framework. Environmental Science and Pollution Research.",
            "Ozbuldu, M. et al. (2025). Projecting and Downscaling Future Temperature and Precipitation Based on CMIP6 Models Using Machine Learning. Pure and Applied Geophysics.",
            "Rampal, N. et al. (2024). Enhancing Regional Climate Downscaling through Advances in Machine Learning. Artificial Intelligence for the Earth Systems.",
            "Météo-France. Description du système SAFRAN et documentation technique associée.",
        ]
        for r in refs:
            ax.text(.075,y,"•",fontsize=9,color=BLUE,va="top"); y=para(ax,r,.098,y,105,6.9,INK,spacing=1.15)
        ax.text(.07,.07,"Note de traçabilité : toutes les métriques numériques proviennent des fichiers de résultats conservés dans le dépôt climate_project.",fontsize=7.5,color=MUTED,style="italic")
        save(fig, pdf)

    print(PDF)


if __name__ == "__main__":
    build()
