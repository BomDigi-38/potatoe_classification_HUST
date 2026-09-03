import argparse
import re
from pathlib import Path
from collections import defaultdict

def analyser_dataset(chemin_original, chemin_traite):
    original = Path(chemin_original)
    traite = Path(chemin_traite)

    if not original.exists() or not traite.exists():
        print("❌ Erreur : L'un des dossiers n'existe pas.")
        return

    print("🔍 Analyse stricte par arborescence en cours...\n")

    # === 1. LECTURE DU DATASET ORIGINAL ===
    # On stocke désormais un couple exact : (nom_base, chemin_relatif) pour éviter les doublons de noms
    original_files = set() 
    categories_original = defaultdict(int)

    for f in original.rglob('*'):
        if f.is_file():
            nom_base = f.stem
            chemin_relatif = str(f.parent.relative_to(original))
            
            original_files.add((nom_base, chemin_relatif))
            categories_original[chemin_relatif] += 1

    # === 2. LECTURE DU DATASET TRAITÉ ===
    treated_counts = defaultdict(int) # Compte les crops pour chaque image stricte
    
    # On sépare les succès des erreurs
    categories_valides = defaultdict(int)
    categories_erreurs = defaultdict(int)
    
    erreurs_par_type = defaultdict(int)

    for f in traite.rglob('*'):
        if f.is_file():
            nom_base = re.sub(r'_t\d+.*$', '', f.stem)
            nom_dossier_parent = f.parent.name.lower()
            est_erreur = "error" in nom_dossier_parent or "errror" in nom_dossier_parent

            # Retrouver le chemin d'origine supposé
            chemin_relatif_traite = f.parent.relative_to(traite)
            if est_erreur:
                chemin_origine = str(chemin_relatif_traite.parent)
            else:
                chemin_origine = str(chemin_relatif_traite)

            # VÉRIFICATION AU DÉTAIL PRÈS : L'image correspond-elle exactement à sa source ?
            if (nom_base, chemin_origine) in original_files:
                treated_counts[(nom_base, chemin_origine)] += 1
                
                # Tri strict entre Valide et Erreur (plus de somme globale)
                if est_erreur:
                    categories_erreurs[chemin_origine] += 1
                    erreurs_par_type[nom_dossier_parent] += 1
                else:
                    categories_valides[chemin_origine] += 1
            else:
                # Si une image est dans le dossier traité mais introuvable dans l'original à cet endroit précis
                # On la compte quand même pour le tableau, mais on sait qu'elle est "orpheline"
                if est_erreur:
                    categories_erreurs[chemin_origine] += 1
                    erreurs_par_type[nom_dossier_parent] += 1
                else:
                    categories_valides[chemin_origine] += 1

    # === 3. CALCUL DES STATISTIQUES ===
    images_manquantes = [f"{c}/{n}" for (n, c) in original_files if (n, c) not in treated_counts]
    images_multipliees = sum(1 for count in treated_counts.values() if count > 1)
    
    total_original = len(original_files)
    total_valides = sum(categories_valides.values())
    total_erreurs = sum(categories_erreurs.values())
    total_crops = total_valides + total_erreurs

    # === 4. AFFICHAGE DU RAPPORT ===
    print("="*85)
    print("📊 RAPPORT D'ANALYSE STRICTE PAR ARBORESCENCE".center(85))
    print("="*85)

    print(f"\n🗂️ VUE D'ENSEMBLE")
    print(f"  - Images originales brutes : {total_original}")
    print(f"  - Images recadrées VALIDES : {total_valides}")
    print(f"  - Images recadrées ERREURS : {total_erreurs}")
    print(f"  - Nombre de dossiers uniques : {len(categories_original)}")

    print(f"\n⚠️ DONNÉES MANQUANTES")
    print(f"  - Images originales sans AUCUN crop : {len(images_manquantes)}")

    print(f"\n✂️ CROPS / MULTIPLICATIONS")
    print(f"  - Images sources ayant généré PLUSIEURS crops : {images_multipliees}")
    if len(treated_counts) > 0:
        print(f"  - En moyenne, 1 image brute traitée donne : {total_crops / len(treated_counts):.2f} crop(s)")

    print(f"\n🚨 ANALYSE DES ERREURS")
    print(f"  - Total des erreurs détectées : {total_erreurs}")
    for type_err, count in sorted(erreurs_par_type.items(), key=lambda x: x[1], reverse=True):
        print(f"      * {type_err} : {count} images")

    print("\n📂 DÉTAIL PAR ARBORESCENCE")
    
    def raccourcir_chemin(chemin, max_len=55):
        return chemin if len(chemin) <= max_len else chemin[:15] + "..." + chemin[-(max_len-18):]

    # Le tableau sépare bien les originales, les réussites (Valides), et les échecs (Erreurs)
    print(f"{'Chemin du dossier':<57} | {'Originales':<10} | {'Valides':<10} | {'Erreurs':<10}")
    print("-" * 95)
    
    for cat in sorted(categories_original.keys()):
        nb_orig = categories_original[cat]
        nb_valide = categories_valides.get(cat, 0)
        nb_err = categories_erreurs.get(cat, 0)
        
        chemin_affichage = raccourcir_chemin(cat)
        print(f"{chemin_affichage:<57} | {nb_orig:<10} | {nb_valide:<10} | {nb_err:<10}")

    print("\n" + "="*85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", help="Dossier original")
    parser.add_argument("traite", help="Dossier traité")
    args = parser.parse_args()
    analyser_dataset(args.original, args.traite)