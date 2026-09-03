import os
import shutil
import argparse
import re
from pathlib import Path

def restaurer_arborescence(chemin_source, chemin_melange, chemin_destination):
    source = Path(chemin_source)
    melange = Path(chemin_melange)
    destination = Path(chemin_destination)

    if not source.exists():
        print(f"❌ Erreur : Le dossier source '{source}' n'existe pas.")
        return
    if not melange.exists():
        print(f"❌ Erreur : Le dossier mélangé '{melange}' n'existe pas.")
        return

    print("🔍 Étape 1 : Cartographie de l'arborescence d'origine...")
    
    # Le dictionnaire stockera : { "nom_sans_extension" : "chemin/relatif" }
    # Exemple : { "10" : "\Potato Disease Dataset\Black Scurf\10" }
    carte_fichiers = {}
    
    for fichier_source in source.rglob('*'):
        if fichier_source.is_file():
            chemin_relatif = fichier_source.relative_to(source).parent
            # On prend le nom du fichier SANS son extension (.jpg, .png...)
            nom_base_original = fichier_source.stem 
            carte_fichiers[nom_base_original] = chemin_relatif

    print(f"✅ {len(carte_fichiers)} fichiers cartographiés depuis la source.")
    print("\n🚀 Étape 2 & 3 : Copie des fichiers traités vers la nouvelle destination...")
    
    fichiers_copies = 0
    fichiers_introuvables = 0

    for fichier_traite in melange.rglob('*'):
        if fichier_traite.is_file():
            # On prend le nom du fichier traité sans extension (ex: '10_t000')
            nom_brut_traite = fichier_traite.stem

            # On nettoie le nom pour enlever les '_t000' ou '_t000_1'
            nom_nettoye = re.sub(r'_t\d+(?:_\d+)?$', '', nom_brut_traite)

            # On cherche ce nom nettoyé dans notre cartographie
            if nom_nettoye in carte_fichiers:
                sous_dossier_cible = destination / carte_fichiers[nom_nettoye]
                sous_dossier_cible.mkdir(parents=True, exist_ok=True)
                
                # On garde le nom complet du fichier traité (10_t000.png) pour la copie
                fichier_cible = sous_dossier_cible / fichier_traite.name
                
                shutil.copy2(fichier_traite, fichier_cible)
                fichiers_copies += 1
            else:
                fichiers_introuvables += 1

    print("\n=== BILAN ===")
    print(f"✅ Fichiers copiés et rangés : {fichiers_copies}")
    if fichiers_introuvables > 0:
        print(f"⚠️ Fichiers ignorés (non trouvés dans la source) : {fichiers_introuvables}")
    print(f"📁 Votre nouveau dataset propre est ici : {destination.resolve()}")

# ==========================================
# GESTION DES PARAMÈTRES EN LIGNE DE COMMANDE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recrée une arborescence en ignorant les extensions et les index _t00x.")
    
    parser.add_argument("source", help="Le dossier modèle (avec la BONNE arborescence)")
    parser.add_argument("melange", help="Le dossier où se trouvent les images traitées")
    parser.add_argument("destination", help="Le dossier de SORTIE où tout sera bien rangé")
    
    args = parser.parse_args()

    restaurer_arborescence(args.source, args.melange, args.destination)