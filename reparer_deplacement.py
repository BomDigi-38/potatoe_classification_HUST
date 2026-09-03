import os
import shutil
import argparse
import re
from pathlib import Path

def reparer_deplacement(chemin_source, chemin_destination):
    source = Path(chemin_source)
    destination = Path(chemin_destination)

    if not source.exists():
        print(f"❌ Erreur : Le dossier source '{source}' n'existe pas.")
        return
    if not destination.exists():
        print(f"❌ Erreur : Le dossier destination '{destination}' n'existe pas.")
        return

    print("🔍 Étape 1 : Cartographie des dossiers cibles...")
    dossiers_existants = {}
    
    for dossier in destination.rglob('*'):
        if dossier.is_dir():
            dossiers_existants[dossier.name] = dossier

    print(f"✅ {len(dossiers_existants)} dossiers trouvés dans l'arborescence cible.")
    print("\n🚀 Étape 2 : Déplacement des images vers leurs dossiers respectifs...")
    
    fichiers_deplaces = 0
    fichiers_orphelins = 0

    for fichier in source.rglob('*'):
        if fichier.is_file():
            nom_original = fichier.stem
            
            # SOLUTION ICI : La nouvelle formule gère '_t000' ET un éventuel '_1' ajouté à la fin
            # Ex: 'Img-1_t000_1' deviendra 'Img-1'
            nom_base = re.sub(r'_t\d+(?:_\d+)?$', '', nom_original)
            
            if nom_base in dossiers_existants:
                dossier_cible = dossiers_existants[nom_base]
                fichier_cible = dossier_cible / fichier.name
                
                # DÉPLACEMENT du fichier
                shutil.move(str(fichier), str(fichier_cible))
                fichiers_deplaces += 1
            else:
                fichiers_orphelins += 1
                # print(f"⚠️ Non trouvé : {fichier.name} (cherché : {nom_base})")

    print("\n=== BILAN ===")
    print(f"✅ Fichiers remis à leur place : {fichiers_deplaces}")
    if fichiers_orphelins > 0:
        print(f"⚠️ Fichiers restants dans la source (pas de dossier correspondant) : {fichiers_orphelins}")

# ==========================================
# GESTION DES PARAMÈTRES EN LIGNE DE COMMANDE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remet des images en vrac dans leurs dossiers respectifs.")
    
    parser.add_argument("source", help="Le dossier où les images sont en VRAC")
    parser.add_argument("destination", help="Le dossier racine qui contient les SOUS-DOSSIERS vides")
    
    args = parser.parse_args()

    print("\n=== DÉMARRAGE DU SAUVETAGE ===")
    print(f"-> Dossier avec images en vrac : {args.source}")
    print(f"-> Dossier avec arborescence   : {args.destination}")
    print("==============================\n")

    reparer_deplacement(args.source, args.destination)