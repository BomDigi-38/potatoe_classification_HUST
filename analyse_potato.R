#!/usr/bin/env Rscript
# ==============================================================================
# analyse_potato.R
#
# Analyse une ou plusieurs images de pomme(s) de terre à partir d'un chemin
# donné en paramètre, avec le package TubAR (Tuber Analysis in R).
#
# Usage :
#   Rscript analyse_potato.R <image.jpg | dossier_images> [dossier_sortie]
#
# Le premier paramètre peut être :
#   - le chemin d'une image JPEG unique, ou
#   - le chemin d'un dossier : toutes les images .jpg/.jpeg qu'il contient
#     (directement à l'intérieur, sans descendre dans les sous-dossiers)
#     sont alors analysées une par une.
#
# Sorties (dans dossier_sortie, "." par défaut) :
#   - result_<AAAAMMJJ_HHMMSS>_potato.csv : UN SEUL fichier, une ligne par
#     tubercule détecté (toutes images confondues), colonnes forme + peau.
#     L'horodatage dans le nom identifie chaque exécution ; écrit ligne par
#     ligne au fur et à mesure (pas seulement à la fin), donc les résultats
#     déjà obtenus sont conservés même si le script plante ou est interrompu
#     en cours de route.
#   - dossier_sortie/tmp_analyse/   : fichiers intermédiaires (image avec
#     correction d'éclairage, pour vérification visuelle en cas de résultat
#     suspect). Ce dossier est entièrement vidé au début de chaque exécution.
#
# Une barre de progression indique l'avancement lorsque plusieurs images
# sont traitées.
#
# Optimisations : chaque image n'est lue qu'une seule fois depuis le disque
# (pas de copie de l'original, pas de relecture du fichier corrigé) ;
# find.shape()/find.skin() ont été patchées pour accepter directement
# l'image déjà chargée en mémoire plutôt que de la relire depuis un fichier.
#
# Prérequis : les images doivent être au format JPEG (contrainte imposée par
# TubAR), idéalement prises sur boîte à lumière avec une carte de couleurs
# (colorcard) dans un coin pour la correction colorimétrique.
# ==============================================================================

## ---- 1. Arguments et résolution des image(s) à traiter ---------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(
    "Usage : Rscript analyse_pomme_de_terre.R <image.jpg | dossier_images> [dossier_sortie]\n",
    call. = FALSE
  )
}

input_path <- args[1]
output_dir <- if (length(args) >= 2) args[2] else "."

if (dir.exists(input_path)) {
  image_paths <- list.files(
    input_path,
    pattern    = "\\.jpe?g$",
    ignore.case = TRUE,
    full.names = TRUE
  )
  if (length(image_paths) == 0) {
    stop(
      sprintf("Aucune image JPEG (.jpg/.jpeg) trouvée dans le dossier '%s'.", input_path),
      call. = FALSE
    )
  }
  cat(sprintf("Dossier détecté : %d image(s) JPEG à analyser.\n\n", length(image_paths)))
} else if (file.exists(input_path)) {
  if (!grepl("\\.jpe?g$", input_path, ignore.case = TRUE)) {
    stop("TubAR nécessite une image au format JPEG (.jpg/.jpeg).", call. = FALSE)
  }
  image_paths <- input_path
} else {
  stop(sprintf("'%s' n'est ni un fichier ni un dossier existant.", input_path), call. = FALSE)
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Dossier des fichiers intermédiaires : entièrement vidé à chaque exécution,
# pour ne jamais accumuler de fichiers d'une exécution à l'autre.
tmp_dir <- file.path(output_dir, "tmp_analyse")
if (dir.exists(tmp_dir)) {
  unlink(tmp_dir, recursive = TRUE, force = TRUE)
}
dir.create(tmp_dir, recursive = TRUE)

## ---- 2. Installation / chargement des dépendances --------------------------

install_if_missing <- function(pkg, bioc = FALSE, github = NULL) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installation du package manquant : %s", pkg))
    if (!is.null(github)) {
      if (!requireNamespace("remotes", quietly = TRUE)) {
        install.packages("remotes", repos = "https://cloud.r-project.org")
      }
      remotes::install_github(github)
    } else if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org")
      }
      BiocManager::install(pkg, update = FALSE, ask = FALSE)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}

install_if_missing("BiocManager")
install_if_missing("EBImage", bioc = TRUE)
install_if_missing("Biobase", bioc = TRUE)
install_if_missing("TubAR", github = "shannonlabumn/TubAR")

suppressPackageStartupMessages({
  library(EBImage)
  library(TubAR)
})

## ---- 2bis. Correctif TubAR::find.shape() (accepte une image en mémoire) ----
# find.shape() d'origine ne prend qu'un chemin de fichier et le relit
# systématiquement (`im <- readImage(image)`). Pour éviter d'écrire l'image
# corrigée sur disque puis de la relire aussitôt (2 accès disque évitables
# par image), cette version accepte aussi un objet Image déjà chargé via le
# nouveau paramètre `im`. Reste identique sinon (copié depuis
# `print(TubAR::find.shape)`), y compris les fonctions internes non
# exportées du package (`TubAR:::...`) et `sp::Polygon`.
find.shape <- function (image = NULL, im = NULL, pix.min = 4000, scaledown = 4,
    colorcard = "bottomright", background.color = "white")
{
    if (is.null(im)) im <- readImage(image)
    im2 <- resize(im, h = dim(im)[2]/scaledown)
    if (scaledown > 1) {
        pix.min <- pix.min/scaledown
    }
    if (background.color == "black") {
        gr <- im2@.Data[, , 1]
        bi <- gr < 0.5
    }
    else if (background.color == "blue") {
        gr <- im2@.Data[, , 3]
        bi <- gr > 0.55
    }
    else {
        gr <- im2@.Data[, , 3]
        bi <- gr > 0.75
    }
    bifil <- fillHull(1 - bi)
    lab <- bwlabel(bifil)
    tab <- data.frame(table(lab))
    tab$lab <- as.numeric(as.character(tab$lab))
    stab <- tab$lab[which(tab$Freq < pix.min)]
    labels <- lab
    labels[which(lab@.Data %in% stab, arr.ind = T)] <- 0
    # Garde-fou : si aucun objet ne survit au filtrage pix.min (ex. tubercule
    # trop petit par rapport au seuil), `max(labels)` vaut 0 et le code
    # d'origine TubAR poursuivrait avec `1:0` (= c(1, 0) en R, pas une
    # séquence vide), ce qui finit par planter plus loin. On renvoie ici un
    # résultat "vide" propre (0 tubercule) plutôt que de laisser planter.
    if (max(labels) == 0) {
        empty <- vector("list", 0)
        return(list(bbox.width = empty, bbox.height = empty, perim = empty,
            convex.perim = empty, area = empty, chull.area = empty,
            roundness = empty, compactness = empty, max.length = empty))
    }
    dist <- X <- Y <- rep(NA, max(labels))
    for (i in 1:max(labels)) {
        ix <- which(labels == i, arr.ind = T)
        X[i] = mean(ix[, 1])
        Y[i] = mean(ix[, 2])
        dist[i] <- sqrt(X[i]^2 + Y[i]^2)
    }
    labels2 <- labels
    if (colorcard == "bottomright") {
        labels2[which(labels == which.max(dist), arr.ind = T)] <- 0
    }
    if (colorcard == "topleft") {
        labels2[which(labels == which.min(dist), arr.ind = T)] <- 0
    }
    if (colorcard == "topright" | colorcard == "bottomleft") {
        X2 <- X - dim(labels)[1] + 1
        dist2 <- sqrt(X2^2 + Y^2)
        if (colorcard == "topright") {
            labels2[which(labels == which.min(dist2), arr.ind = T)] <- 0
        }
        if (colorcard == "bottomleft") {
            labels2[which(labels == which.max(dist2), arr.ind = T)] <- 0
        }
    }
    p <- unique(c(labels2))
    p <- setdiff(p, c(0))
    bbox.width <- bbox.height <- perimeter <- convex.perim <- area <- chull.area <- roundness <- compactness <- max.length <- ends <- vector("list",
        length(p))
    labelsX <- medianFilter(labels2, 50/scaledown)
    for (i in 1:length(p)) {
        labelsx <- labelsX
        labelsx[which(labels2 != p[i])] <- 0
        outline <- TubAR:::get.trace(labelsx)
        perimeter[[i]] <- sum(outline > 0)
        area[[i]] <- sum(labelsx)
        outline2 <- which(outline > 0, arr.ind = T)
        dist <- as.matrix(dist(outline2))
        max.length[[i]] <- max(dist)
        compactness[[i]] <- (4 * pi * area[[i]])/(perimeter[[i]]^2)
        ic <- chull(outline2)
        convexhull <- outline2[c(ic, ic[1]), ]
        convex.perim[[i]] <- TubAR:::get.perim(convexhull)
        chull.poly <- sp::Polygon(convexhull, hole = F)
        chull.area[[i]] <- chull.poly@area
        roundness[[i]] <- (4 * pi * chull.area[[i]])/(convex.perim[[i]]^2)
        box <- TubAR:::getMinBBox(outline2)
        bbox.width[[i]] <- min(box$width, box$height)
        bbox.height[[i]] <- max(box$width, box$height)
    }
    return(list(bbox.width = bbox.width, bbox.height = bbox.height,
        perim = perimeter, convex.perim = convex.perim, area = area,
        chull.area = chull.area, roundness = roundness, compactness = compactness,
        max.length = max.length))
}

## ---- 2ter. Correctif TubAR::find.skin() -------------------------------------
# Deux correctifs par rapport à l'original (copié depuis
# `print(TubAR::find.skin)`) :
#   1. Accepte aussi une image déjà chargée en mémoire via le paramètre `im`
#      (même logique que find.shape ci-dessus, évite une relecture disque).
#   2. Bug TubAR 1.1.0 : quand un seul tubercule subsiste après filtrage,
#      `rgb <- sapply(p, function(x) cbind(...))` (p de longueur 1) renvoie
#      directement la matrice "aplatie" au lieu d'une liste d'1 élément
#      (sapply() ne "simplifie" ainsi que lorsqu'il y a un seul appel). Le
#      code suivant traite alors les colonnes de cette matrice comme des
#      objets séparés, ce qui casse tout en aval ("dim(X) doit avoir une
#      longueur positive"). Remplacé par lapply(), qui renvoie toujours une
#      liste quel que soit le nombre d'objets détectés.
#
# NB : le chemin `color.correct = TRUE` (carte de couleurs) n'a pas été
# testé avec ce correctif ; ce script utilise `color_correct <- FALSE`. Si
# `image` n'est pas fourni (seulement `im`), ce chemin plantera (nécessite
# `image` pour `grabcard()`), de même que `write.clean = TRUE`.
find.skin <- function (image = NULL, im = NULL, display = T, mode = "debug", write.clean = F,
    pix.min = 40000, scaledown = 8, colorcard = "bottomright",
    n.core = 1, color.correct = T, background.color = "white",
    color.values = "revised", color.center = "default")
{
    if (is.null(im)) im <- readImage(image)
    im2 <- resize(im, h = dim(im)[2]/scaledown)
    if (scaledown > 1) {
        pix.min <- pix.min/scaledown
    }
    if (background.color == "black") {
        gr <- im2@.Data[, , 1]
        bi <- gr < 0.5
    }
    else if (background.color == "blue") {
        gr <- im2@.Data[, , 3]
        bi <- gr > 0.55
    }
    else {
        gr <- im2@.Data[, , 3]
        bi <- gr > 0.75
    }
    bifil <- fillHull(1 - bi)
    lab <- bwlabel(bifil)
    tab <- data.frame(table(lab))
    tab$lab <- as.numeric(as.character(tab$lab))
    stab <- tab$lab[which(tab$Freq < pix.min)]
    labels <- lab
    labels[which(lab@.Data %in% stab, arr.ind = T)] <- 0
    # Garde-fou : mêmes raisons que dans find.shape() ci-dessus (voir son
    # commentaire) - si aucun objet ne survit au filtrage pix.min, on
    # renvoie un résultat vide plutôt que de laisser planter plus loin.
    if (max(labels) == 0) {
        return(list(skinning = numeric(0), redness = numeric(0), lightness = numeric(0)))
    }
    dist <- X <- Y <- rep(NA, max(labels))
    for (i in 1:max(labels)) {
        ix <- which(labels == i, arr.ind = T)
        X[i] = mean(ix[, 1])
        Y[i] = mean(ix[, 2])
        dist[i] <- sqrt(X[i]^2 + Y[i]^2)
    }
    labels2 <- labels
    if (colorcard == "bottomright") {
        labels2[which(labels == which.max(dist), arr.ind = T)] <- 0
    }
    if (colorcard == "topleft") {
        labels2[which(labels == which.min(dist), arr.ind = T)] <- 0
    }
    if (colorcard == "topright" | colorcard == "bottomleft") {
        X2 <- X - dim(labels)[1] + 1
        dist2 <- sqrt(X2^2 + Y^2)
        if (colorcard == "topright") {
            labels2[which(labels == which.min(dist2), arr.ind = T)] <- 0
        }
        if (colorcard == "bottomleft") {
            labels2[which(labels == which.max(dist2), arr.ind = T)] <- 0
        }
    }
    p <- unique(c(labels2))
    p <- setdiff(p, c(0))
    rgb <- lapply(p, function(x) cbind(im2@.Data[, , 1][which(labels2 ==
        x)], im2@.Data[, , 2][which(labels2 == x)], im2@.Data[,
        , 3][which(labels2 == x)]))
    if (color.correct == T) {
        obs.land <- TubAR:::grabcard(image, colorcard = colorcard, scaledown = scaledown,
            pix.min = pix.min, color.center = color.center)
        if (length(obs.land) == 72) {
            if (color.values == "default") {
                card <- matrix(c(116, 81, 67, 199, 147, 129,
                  91, 122, 156, 90, 108, 64, 130, 128, 176, 92,
                  190, 172, 224, 124, 47, 68, 91, +170, 198,
                  82, 97, 94, 58, 106, 159, 189, 63, 230, 162,
                  39, 34, 63, 147, 67, 149, 74, 180, 49, 57,
                  238, 198, +32, 193, 84, 151, 12, 136, 170,
                  243, 238, 243, 200, 202, 202, 161, 162, 161,
                  120, 121, 120, 82, 83, 83, 49, 48, 51), nrow = 24,
                  ncol = 3, byrow = T)
                card2 <- as.matrix(card/255)
            }
            else if (color.values == "revised") {
                card2 <- matrix(c(0.4823529, 0.4156863, 0.4784314,
                  0.9019608, 0.7058824, 0.7490196, 0.5058824,
                  0.6627451, +0.9058824, 0.3490196, 0.5960784,
                  0.5137255, 0.6823529, 0.7019608, 0.9607843,
                  0.6431373, +0.9372549, 0.9568627, 0.9254902,
                  0.5803922, 0.4, 0.3960784, 0.4901961, 0.9058824,
                  +0.8352941, 0.3803922, 0.5568627, 0.3607843,
                  0.2666667, 0.6117647, 0.7254902, 0.9176471,
                  +0.5607843, 0.9647059, 0.7647059, 0.4470588,
                  0.254902, 0.3058824, 0.7058824, 0.3411765,
                  +0.7764706, 0.6078431, 0.7098039, 0.2313725,
                  0.3215686, 0.9803922, 0.9137255, 0.4666667,
                  +0.8705882, 0.4588235, 0.854902, 0.4235294,
                  0.7215686, 0.9568627, 0.9764706, 0.9529412,
                  +0.9607843, 0.8862745, 0.9137255, 0.9372549,
                  0.7568627, 0.8196078, 0.8823529, 0.5490196,
                  +0.6627451, 0.7960784, 0.3333333, 0.4509804,
                  0.5843137, 0.2784314, 0.3333333, 0.4470588),
                  nrow = 24, ncol = 3, byrow = T)
            }
            else {
                card <- color.values
                card2 <- as.matrix(card/255)
            }
            rgb <- lapply(rgb, function(x) Morpho::tps3d(x, obs.land,
                card2))
        }
        else {
            warning("Error: color correction failure. Is card crooked?")
        }
    }
    if (n.core > 1) {
        future::plan(future::multiprocess, workers = n.core)
        Lab <- future.apply::future_lapply(rgb, function(x) t(apply(x, 1, function(y) convertColor(y,
            from = "sRGB", to = "Lab"))))
    }
    else {
        Lab <- lapply(rgb, function(x) t(apply(x, 1, function(y) convertColor(y,
            from = "sRGB", to = "Lab"))))
    }
    thresh <- vector("list", length(p))
    names(thresh) <- p
    skinper <- vector("list", length(p))
    names(thresh) <- p
    for (i in 1:length(p)) {
        thresh[[i]] <- c(min(Lab[[i]][, 3], na.rm = T):max(Lab[[i]][,
            3], na.rm = T))
        skinper[[i]] <- sapply(thresh[[i]], function(x) sum(Lab[[i]][,
            3] > x)/(dim(Lab[[i]])[1]))
    }
    curve <- data.frame(x = unlist(thresh), y = unlist(skinper))
    sigm <- minpack.lm::nlsLM(y ~ a/(1 + exp(-b * (x - c))) + d, data = curve,
        start = list(a = max(curve$y), b = 1, c = median(curve$x),
            d = min(curve$y)))
    c = summary(sigm)$parameters["c", "Estimate"]
    thr <- c * 1.5
    skinpot <- sapply(1:length(p), function(i) sum(Lab[[i]][,
        3] > thr)/(dim(Lab[[i]])[1]))
    red.intens <- sapply(1:length(p), function(i) median(Lab[[i]][,
        2][which(Lab[[i]][, 3] < thr)]))
    lightness <- sapply(1:length(p), function(i) median(Lab[[i]][,
        1][which(Lab[[i]][, 3] < thr)]))
    labels3 <- labels2
    labels3[which(labels3 > 0, arr.ind = T)] <- 1
    template <- abs(labels3 - 1)
    if (display == T) {
        for (i in 1:length(p)) {
            template[which(labels2 == p[i])[which(Lab[[i]][,
                3] > thr)]] <- 0.5
        }
        display(template, "raster")
        if (mode == "debug") {
            L <- setdiff(unique(c(labels2)), 0)
            for (i in L) {
                ix <- which(labels == i, arr.ind = T)
                text(x = mean(ix[, 1]), y = mean(ix[, 2]), labels = which(L %in%
                  i), cex = 2, col = "lightblue")
            }
        }
    }
    if (write.clean == T) {
        test <- resize(template, h = dim(im2)[2] * scaledown)
        clean.im <- im
        R <- clean.im@.Data[, , 1]
        R[which(test@.Data == 1, arr.ind = T)] <- 1
        G <- clean.im@.Data[, , 2]
        G[which(test@.Data == 1, arr.ind = T)] <- 1
        B <- clean.im@.Data[, , 3]
        B[which(test@.Data == 1, arr.ind = T)] <- 1
        clean.im@.Data[, , 1] <- R
        clean.im@.Data[, , 2] <- G
        clean.im@.Data[, , 3] <- B
        writeImage(clean.im, paste0("clean_", image))
    }
    result <- list(skinning = round(skinpot, 2), redness = round(red.intens,
        1), lightness = round(lightness, 1))
    return(result)
}

## ---- 3. Paramètres d'analyse ------------------------------------------------
# Ajuste ces valeurs selon ton dispositif de prise de vue si besoin :
#   - colorcard   : coin où se trouve la carte de couleurs ("bottomright",
#                   "bottomleft", "topright", "topleft", ou NULL si absente)
#   - pix.min     : nb de pixels min pour qu'un objet soit considéré comme
#                   un tubercule (permet d'ignorer poussière/débris)
#   - scaledown   : facteur de réduction de l'image pour accélérer le calcul

# NB : mets colorcard <- "bottomright" (ou "bottomleft"/"topright"/"topleft")
# UNIQUEMENT si tes photos contiennent réellement une carte de couleurs de
# calibration (24 cases colorées) dans ce coin.
#
# ATTENTION - bug connu de TubAR 1.1.0 : la doc dit que colorcard = NULL
# désactive la détection de carte, mais le code source fait
# `if(colorcard=="bottomright")`, et NULL=="bottomright" vaut logical(0),
# ce qui fait planter if() avec "argument is of length zero". On utilise
# donc une valeur sentinelle ("none") qui ne correspond à aucun des 4 coins
# reconnus : toutes les comparaisons deviennent FALSE (pas d'erreur) et
# aucun objet n'est retiré de l'image - exactement le comportement voulu
# quand il n'y a pas de vraie carte physique dans la photo.
colorcard <- "none"
color_correct <- FALSE
pix_min_shape <- 4000
scaledown_shape <- 4
scaledown_skin  <- 8

# pix_min_skin est calculé à partir de pix_min_shape (et non fixé à part,
# comme le fait TubAR par défaut avec 40000) pour qu'un tubercule qui passe
# le filtre de find.shape() passe aussi celui de find.skin().
#
# TubAR compare `pix.min / scaledown` à l'aire en pixels DANS L'IMAGE
# RÉDUITE ; l'aire réelle minimale retenue équivaut donc à
# `pix.min * scaledown`. Avec les valeurs par défaut de TubAR
# (pix.min=4000/scaledown=4 pour la forme, pix.min=40000/scaledown=8 pour
# la peau), cette aire réelle minimale valait 16000 pour la forme contre
# 320000 pour la peau (20x plus stricte) : un petit tubercule pouvait donc
# être détecté par find.shape() mais entièrement filtré par find.skin(),
# jusqu'à ne plus laisser aucun objet - et faire planter find.skin() (voir
# le garde-fou ci-dessus). En égalant les deux aires réelles minimales :
#   pix_min_skin * scaledown_skin = pix_min_shape * scaledown_shape
pix_min_skin <- pix_min_shape * scaledown_shape / scaledown_skin

# Affichage détaillé (TRUE) : imprime pour chaque image le détail des
# résultats (forme + peau). Par défaut FALSE : seule la barre de progression
# et le résumé final s'affichent, pour un rendu propre en traitement par lot.
VERBOSE <- FALSE

## ---- 3bis. Correction d'éclairage du fond (flat-field) ---------------------
# TubAR détecte le fond via un seuil fixe sur le canal bleu (> 0.75 en mode
# background.color = "white"). Un éclairage non uniforme (ombres portées,
# vignettage de l'objectif) peut faire passer de grandes zones de fond
# sous ce seuil : elles sont alors comptées à tort comme des "tubercules"
# supplémentaires (observé : jusqu'à ~68 % de l'image classée à tort comme
# premier plan sur une photo pourtant prise sur fond clair uniforme).
#
# Correction de type "champ plat" (flat-field) : on estime l'éclairage du
# fond par fermeture morphologique en niveaux de gris (efface les zones
# sombres/tubercules tout en gardant le fond), puis on ramène chaque pixel
# vers une luminosité de fond cible uniforme. Le tubercule, nettement plus
# sombre que le fond, reste bien en dessous du seuil après correction.
#
# ATTENTION : approche empirique, rapide à mettre en place mais fragile si
# l'exposition/l'éclairage varie beaucoup d'une photo à l'autre, ou si le
# sujet occupe une part très différente du cadre. En cas de résultats
# suspects (nombre de tubercules détectés incohérent), vérifier l'image
# corrigée dans tmp_analyse/<nom_image>_corrige.jpg.

correct_illumination <- function(im, target = 0.92, ds_width = 400, brush_frac = 0.55) {
  ds  <- resize(im, w = ds_width)
  lum <- 0.299 * ds@.Data[, , 1] + 0.587 * ds@.Data[, , 2] + 0.114 * ds@.Data[, , 3]
  brush_size <- max(3, round(brush_frac * min(dim(lum)[1:2])))
  if (brush_size %% 2 == 0) brush_size <- brush_size + 1
  bg <- closing(lum, makeBrush(brush_size, shape = "disc"))
  bg[bg < 0.05] <- 0.05  # évite une division par (quasi) zéro
  factor_full <- resize(target / bg, w = dim(im)[1], h = dim(im)[2])
  corrected <- im
  for (ch in 1:3) {
    corrected@.Data[, , ch] <- pmin(pmax(im@.Data[, , ch] * factor_full, 0), 1)
  }
  corrected
}

## ---- 3ter. Barre de progression ---------------------------------------------
# Barre personnalisée (blocs Unicode, %, ETA, nom du fichier en cours),
# réécrite en place sur une seule ligne via un retour chariot ("\r") plutôt
# qu'une nouvelle ligne à chaque mise à jour. Reste stable/lisible tant que
# rien d'autre n'est imprimé entre deux mises à jour : voir VERBOSE ci-dessus,
# qui coupe par défaut les impressions détaillées par image pour cette
# raison (la barre de progression est alors le seul retour visuel pendant le
# traitement).

draw_progress_bar <- function(current, total, label, start_time, width = 100) {
  frac <- current / total
  filled <- round(frac * width)
  bar <- paste0(strrep("█", filled), strrep(" ", width - filled))
  elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
  eta_sec <- if (current > 0) (elapsed / current) * (total - current) else NA
  eta_str <- if (is.na(eta_sec) || !is.finite(eta_sec)) {
    "--:--"
  } else {
    sprintf("%02d:%02d", eta_sec %/% 60, round(eta_sec %% 60))
  }
  label_trunc <- if (nchar(label) > 30) paste0(substr(label, 1, 27), "...") else label
  cat(sprintf(
    "\r[%s] %3d%%  %d/%d  ETA %s  %-30s",
    bar, round(frac * 100), current, total, eta_str, label_trunc
  ))
  utils::flush.console()
  if (current >= total) cat("\n")
}

## ---- 4. Analyse d'une image -------------------------------------------------
# Traite une seule image (affichage interactif, correction d'éclairage,
# find.shape, find.skin) et renvoie un data.frame combinant forme + peau
# (une ligne par tubercule détecté), ou NULL si rien d'exploitable.
#
# Chaque image n'est lue qu'une fois (readImage) ; find.shape()/find.skin()
# réutilisent directement cette image en mémoire (paramètre `im`), sans
# jamais relire depuis le disque. Seule l'image corrigée est écrite sur
# disque, dans tmp_dir, à titre de vérification visuelle (pas nécessaire à
# l'analyse elle-même).

analyser_image <- function(image_path, tmp_dir) {

  base_name <- tools::file_path_sans_ext(basename(image_path))

  img <- tryCatch(
    readImage(image_path),
    error = function(e) {
      warning(sprintf("Impossible de lire '%s' : %s", image_path, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(img)) {
    return(NULL)
  }

  # Affichage interactif (fenêtre graphique / RStudio Viewer) : ignoré
  # silencieusement si aucune session graphique n'est disponible (ex.
  # Rscript exécuté en ligne de commande/batch sans interface). Ne touche
  # pas au disque.
  tryCatch(
    {
      display(img, method = "raster")
      title(main = basename(image_path))
    },
    error = function(e) message("Affichage interactif indisponible : ", conditionMessage(e))
  )

  # Correction d'éclairage du fond (en mémoire) ; en cas d'échec, on retombe
  # sur l'image d'origine pour ne pas bloquer l'analyse.
  analysis_img <- tryCatch(correct_illumination(img), error = function(e) {
    warning(sprintf("Échec de la correction d'éclairage sur '%s' : %s", image_path, conditionMessage(e)))
    NULL
  })
  if (is.null(analysis_img)) {
    analysis_img <- img
  } else {
    tryCatch(
      writeImage(analysis_img, file.path(tmp_dir, paste0(base_name, "_corrige.jpg")), quality = 95),
      error = function(e) warning(sprintf("Échec d'écriture de l'image corrigée pour '%s' : %s", image_path, conditionMessage(e)))
    )
  }

  shape_result <- tryCatch(
    find.shape(im = analysis_img, pix.min = pix_min_shape, scaledown = scaledown_shape, colorcard = colorcard),
    error = function(e) {
      warning(sprintf("Échec de find.shape() sur '%s' : %s", image_path, conditionMessage(e)))
      NULL
    }
  )

  skin_result <- tryCatch(
    find.skin(
      im            = analysis_img,
      display       = TRUE,
      mode          = "debug",
      write.clean   = FALSE,
      pix.min       = pix_min_skin,
      scaledown     = scaledown_skin,
      colorcard     = colorcard,
      n.core        = 1,
      color.correct = color_correct
    ),
    error = function(e) {
      warning(sprintf("Échec de find.skin() sur '%s' : %s", image_path, conditionMessage(e)))
      NULL
    }
  )

  # --- Mise en forme : un data.frame par analyse, puis fusion en une ligne
  # par tubercule (forme + peau) ---

  # unlist(list()) renvoie NULL plutôt que numeric(0) ; as.data.frame() sur
  # une colonne NULL la fait disparaître entièrement (au lieu d'une colonne
  # vide), ce qui casserait la sélection de colonnes plus bas quand 0
  # tubercule est détecté. On force donc numeric(0) dans ce cas.
  unlist_or_empty <- function(x) {
    u <- unlist(x)
    if (is.null(u)) numeric(0) else u
  }

  shape_df <- NULL
  if (!is.null(shape_result)) {
    shape_df <- tryCatch({
      cols <- lapply(shape_result, unlist_or_empty)
      n_obj <- max(vapply(cols, length, integer(1)))
      df <- as.data.frame(lapply(cols, function(x) {
        length(x) <- n_obj  # complète avec NA si des traits ont des longueurs différentes
        x
      }))
      cbind(tuber_id = seq_len(nrow(df)), df)
    }, error = function(e) NULL)
  }

  skin_df <- NULL
  if (!is.null(skin_result)) {
    skin_df <- tryCatch({
      if (!is.null(skin_result$by.tuber)) {
        bt <- skin_result$by.tuber
        d <- tryCatch(do.call(rbind, lapply(bt, as.data.frame)), error = function(e) NULL)
        if (is.null(d)) {
          cols <- lapply(bt, unlist_or_empty)
          n_obj <- max(vapply(cols, length, integer(1)))
          d <- as.data.frame(lapply(cols, function(x) { length(x) <- n_obj; x }))
        }
        d
      } else {
        as.data.frame(skin_result)
      }
    }, error = function(e) NULL)
    if (!is.null(skin_df)) {
      skin_df <- cbind(tuber_id = seq_len(nrow(skin_df)), skin_df)
    }
  }

  shape_cols <- c("bbox.width", "bbox.height", "perim", "convex.perim", "area", "chull.area", "roundness", "compactness", "max.length")
  skin_cols  <- c("skinning", "redness", "lightness")

  combined <- NULL
  if (!is.null(shape_df)) {
    combined <- shape_df[, c("tuber_id", shape_cols), drop = FALSE]
  } else if (!is.null(skin_df)) {
    combined <- data.frame(tuber_id = skin_df$tuber_id)
    combined[shape_cols] <- NA_real_
  } else if (VERBOSE) {
    cat("Aucun résultat exploitable (ni forme, ni peau) pour cette image.\n")
  }

  if (!is.null(combined) && nrow(combined) == 0) {
    # 0 tubercule survivant au filtrage pix.min (voir garde-fou dans
    # find.shape()/find.skin() ci-dessus) : rien à fusionner ni à exporter
    # pour cette image, mais ce n'est pas une erreur en soi.
    if (VERBOSE) {
      cat("0 tubercule exploitable pour cette image (sous les seuils pix.min).\n")
    }
    combined <- NULL
  }

  if (!is.null(combined)) {
    if (!is.null(skin_df) && nrow(skin_df) == nrow(combined)) {
      combined <- cbind(combined, skin_df[, skin_cols, drop = FALSE])
    } else {
      if (!is.null(skin_df)) {
        warning(sprintf(
          "'%s' : nombre de tubercules différent entre find.shape (%d) et find.skin (%d) ; colonnes de peau non fusionnées.",
          basename(image_path), nrow(combined), nrow(skin_df)
        ))
      }
      combined[skin_cols] <- NA_real_
    }
    combined <- cbind(image = basename(image_path), combined)
    if (VERBOSE) {
      cat("Résultats (forme + peau) :\n")
      print(combined)
    }
  }

  combined
}

## ---- 5. Boucle sur la ou les image(s) et export du CSV final ---------------
# Le CSV est écrit ligne par ligne au fur et à mesure (append), et non
# assemblé puis écrit une seule fois à la toute fin : si le script plante ou
# est interrompu en cours de route (erreur imprévue, Ctrl+C, coupure...),
# toutes les images déjà traitées avec succès restent dans le fichier.

n_images <- length(image_paths)
all_results <- vector("list", n_images)

# Horodatage dans le nom du fichier : chaque exécution produit son propre
# CSV (pas de risque d'écraser le résultat d'une exécution précédente).
run_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
csv_path <- file.path(output_dir, sprintf("result_%s_potato.csv", run_timestamp))
header_written <- FALSE

cat(sprintf("Traitement de %d image(s)...\n", n_images))
start_time <- Sys.time()

for (i in seq_along(image_paths)) {
  img <- image_paths[i]
  if (VERBOSE && n_images > 1) {
    cat(sprintf("\n########## [%d/%d] %s ##########\n\n", i, n_images, basename(img)))
  }
  combined <- tryCatch(
    analyser_image(img, tmp_dir),
    error = function(e) {
      warning(sprintf("Échec du traitement de '%s' : %s", img, conditionMessage(e)))
      NULL
    }
  )
  all_results[[i]] <- combined

  if (!is.null(combined)) {
    write.table(
      combined, file = csv_path, sep = ",", quote = FALSE, row.names = FALSE,
      col.names = !header_written, append = header_written
    )
    header_written <- TRUE
  }

  draw_progress_bar(i, n_images, basename(img), start_time)
}

all_results <- all_results[!vapply(all_results, is.null, logical(1))]

cat("\n")
if (length(all_results) == 0) {
  cat("Aucun résultat exploitable sur l'ensemble des images traitées ; pas de CSV généré.\n")
} else {
  final_df <- do.call(rbind, all_results)
  cat(sprintf("%d ligne(s) (tubercule(s)) exportée(s) vers : %s\n", nrow(final_df), csv_path))
  cat("Aperçu :\n")
  print(final_df)
}

cat(sprintf("\n%d/%d image(s) traitée(s) avec un résultat exploitable.\n", length(all_results), n_images))
cat("Analyse terminée.\n")



