# Omni-FR — plan d'exécution

Cible : 500 M de paramètres, Raspberry Pi 5 (8 Go) ou Mac M1/M2, **achetés seulement après
validation**. Entraînement sur le PC (RTX 5000 Ada 32 Go, 128 Go de RAM). Les durées sont en
semaines à plein temps ; les seuils des portes sont des hypothèses de départ, à recaler.

## Résultats déjà acquis (12 novembre 2026)

| Mesure | Résultat | Où |
| --- | --- | --- |
| Formes des bases candidates | FunctionGemma 270M = 100,3 M hors embeddings ; Qwen3-0.6B = 440,5 M ; Qwen3-0.6B élaguée à 16 couches = 251,7 M | `omnifr/config.py`, conformes aux cartes publiques |
| Streaming = hors ligne | Écart nul entre pas à pas (caches) et calcul parallèle, backbone natif et HF | `tests/test_model.py` |
| Fenêtre de l'encodeur | 24 trames de mel, pas 16 | `AudioEncoder.step_window` |
| Décodage tronqué du codec Qwen | Identique au décodeur officiel à 16 codebooks (écart 0,0) | test ponctuel, poids aléatoires |
| Étudiant omni500, 8 codebooks, int8 | 73 à 84 ms par trame sur 2 cœurs Xeon 2,3 GHz, soit ~100 % du temps réel | `bench/bench_step.py` |
| Décodeur codec Qwen, config par défaut | 195 M ; 2,7× le temps réel en blocs de 320 ms ; 0,8× en blocs de 2 s | `bench/bench_codec.py --random` |
| Décodeur Mimi, mêmes conditions | ~40 M ; 0,51× en blocs de 320 ms ; 0,15× en blocs de 2 s | même protocole |

Lecture : sur CPU, **le décodeur du codec Qwen est un risque plus grave que l'étudiant**. Les
deux mesures codec recalculent 25 trames de contexte à chaque bloc ; un décodeur à état ferait
mieux, mais Mimi en a déjà un (Moshi l'utilise en temps réel), pas le décodeur Qwen.

## Semaine 1 — Valider le matériel sans l'acheter

- [ ] Installer et lancer `python -m pytest -q` sur le PC Windows.
- [ ] Publier le dépôt sur GitHub (public), lancer le workflow `bench-arm` 3 fois : Mac M1 (`macos-14`) et Linux arm64.
- [ ] Ouvrir une instance gratuite Oracle Cloud Ampere A1 (4 cœurs Neoverse N1, la famille dont dérive le Cortex-A76 du Pi 5) et y lancer les deux bancs avec `--threads 4`.
- [ ] Étalonner Oracle → Pi 5 : sur Oracle, `llama-bench` avec Gemma 3 1B Q4_K_M ; le Pi 5 en fait ~11,5 jetons/s (banc publié). Le rapport des deux donne un facteur de correction approximatif.
- [ ] `bench_codec.py` sur le **vrai** checkpoint Qwen (taille réelle du décodeur), puis Mimi.
- [ ] `scripts/codec_roundtrip.py` sur 60 extraits français (Common Voice, MLS) + votre voix de référence, pour Qwen 16/8 codebooks et Mimi 8/16.

**Porte G1 — achat.** Étudiant (int8, 8 codebooks) + décodeur du codec ≤ 70 % du temps réel,
estimé pour le Pi 5. Sinon, Mac M1/M2 si le runner `macos-14` passe le même seuil. Sinon,
réduire (backbone 16 couches, 4 codebooks) et remesurer avant d'acheter quoi que ce soit.

**Porte G2 — codec.** Écart de WER < 1 point par rapport à l'original, voix reconnaissable à
l'écoute à l'aveugle. Si Qwen et Mimi passent tous deux, prendre celui qui passe G1.

## Semaine 2 — Base du backbone et jeu de test

- [ ] Étendre `data/scenarios_fr_seed.jsonl` à 300 scénarios de test, jamais utilisés à l'entraînement.
- [ ] Exporter les vrais schémas MCP de votre Home Assistant et remplacer ceux de `omnifr/sim/home.py`.
- [ ] Enregistrer 100 commandes réelles chez vous (téléphone suffit), à 1 m, 3 m et pièce voisine.
- [ ] Mesurer la cascade actuelle sur le jeu de test : latence, réussite d'outils. C'est la référence.
- [ ] Vérifier si FunctionGemma partage exactement le vocabulaire de Gemma 4, et Qwen3-0.6B celui de Qwen3.6.
- [ ] Duel texte seul : FunctionGemma 270M contre Qwen3-0.6B élaguée à 16 couches, chacune réglée sur ~2 000 dialogues du simulateur générés par l'enseignant texte.

**Porte G3 — base.** Garder la base qui réussit le mieux le jeu de test texte, surtout les
appels parallèles et les confirmations. Si les deux échouent nettement, un 500 M ne fera pas
mieux en parlant : passer à l'option frontend-backend (section 14 bis de l'étude).

## Semaines 3 à 4 — Usine à données v1

- [ ] Générateur : enseignant texte × simulateur → dialogues avec appels, résultats réels, erreurs, interruptions.
- [ ] Voix de l'assistant : TTS retenu, voix clonée ; codes encodés dans le codec choisi en G2.
- [ ] Voix des utilisateurs : nombreuses voix, augmentation bruit/réverbération ; plus Common Voice et MLS réels pour l'écoute.
- [ ] Filtre ASR + aligneur forcé (horodatage mot à mot pour `align_text_to_frames`).
- [ ] Cible : 200 h validées, taux de rejet mesuré.

## Semaines 5 à 8 — Entraînement par étapes

Toutes les ablations sur un modèle d'environ 100 M et 200 h ; le 500 M seulement pour les
recettes gagnantes. Étapes et portes : section 7 de l'étude (écoute, parole, outils en texte
seul, parole à parole, interruption et bruit).

## Semaines 9 à 12 — Temps réel sur l'appareil acheté

Moteur d'inférence (départ : backends MLX ou Rust de Moshi), quantification 4/8 bits entraînée,
annulation d'écho, détection de voix, décodage contraint des appels, orchestrateur MCP.
Porte finale : une semaine d'usage quotidien sans action exécutée à tort.
