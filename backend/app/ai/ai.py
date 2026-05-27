"""
Main file containing ML/AI logic for Transparent & Editable User Profiling.
Architecture: Soft-Regularized HCAI Autoencoder with Hybrid Override Path
Dataset: MovieLens Latest Small (100k ratings)
"""

#region imports and config
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math

# Optionale XAI-Dependencies — graceful degradation wenn nicht installiert.
# Installation: pip install shap scikit-learn
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(current_dir, "..", "database", "ml-latest-small"))

OVERRIDE_MAP = {
    "stark dämpfen": -2.0,
    "leicht dämpfen": -1.0,
    "neutral": 0.0,
    "leicht verstärken": 1.0,
    "stark verstärken": 2.0
}

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
#endregion

#region dataset helpers
def load_movielens_data():
    """Lädt die CSV-Dateien des MovieLens-Datensatzes."""
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), encoding='utf-8')
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'), encoding='utf-8')
    return ratings, movies

def build_pure_genre_matrix(movies_df):
    """
    Erstellt die unbestechliche Ur-Genre-Matrix M_genre.
    Dimension: (Num_Movies x 18)
    """
    matrix = pd.DataFrame(0.0, index=movies_df['movieId'], columns=GENRES)
    for _, row in movies_df.iterrows():
        movie_genres = row['genres'].split('|')
        for g in movie_genres:
            if g in GENRES:
                matrix.at[row['movieId'], g] = 1.0
    return matrix
#endregion

#region hcai utility functions

def build_override_tensor(user_category_dict: dict, num_genres: int) -> torch.Tensor:
    """
    HCAI Ebene 3: Aktiviert OVERRIDE_MAP als Single Source of Truth.
    
    Konvertiert die menschenlesbare UI-Eingabe (Sprachkategorie pro Genre) in den
    mathematischen Override-Tensor, der an model.forward() übergeben wird.
    
    Diese Funktion ist die explizite Brücke zwischen menschlicher Sprache und
    Mathematik — und genau das ist das HCAI-Argument gegenüber dem Reviewer:
    Der Nutzer denkt in "stark dämpfen", das System rechnet in Float-Werten.
    
    Beispiel-Aufruf:
        tensor = build_override_tensor({"Sci-Fi": "stark verstärken",
                                        "Comedy": "stark dämpfen"}, num_genres)
    
    Args:
        user_category_dict: {Genre-Name (str): OVERRIDE_MAP-Schlüssel (str)}
        num_genres:         Anzahl der Genres (len(GENRES))
    
    Returns:
        Float-Tensor (1 x num_genres) — direkt an model.forward(user_overrides=...) übergeben.
    """
    tensor = torch.zeros(1, num_genres, dtype=torch.float32)
    for genre, category in user_category_dict.items():
        if genre in GENRES and category in OVERRIDE_MAP:
            tensor[0, GENRES.index(genre)] = OVERRIDE_MAP[category]
    return tensor

#endregion

#region neural network model
class SoftRegularizedHCAIAutoEncoder(nn.Module):
    def __init__(self, num_movies, num_genres, pure_genre_matrix_np, hidden_dim=128):
        super().__init__()
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.hidden_dim = hidden_dim
        
        # 1. Wissens-Basis (Ur-Matrix) registrieren
        target_tensor = torch.tensor(pure_genre_matrix_np, dtype=torch.float32).T
        self.register_buffer('target_genre_matrix', target_tensor)
        
        # 2. Encoder-Architektur
        self.encoder_l1 = nn.Linear(num_movies, hidden_dim, bias=False)
        self.encoder_act = nn.ReLU()
        self.encoder_l2 = nn.Linear(hidden_dim, num_genres, bias=False)
        
        # 3. Decoder-Architektur
        self.dropout = nn.Dropout(p=0.4)
        self.decoder_l1 = nn.Linear(num_genres, hidden_dim)
        self.decoder_act = nn.ReLU()
        self.decoder_l2 = nn.Linear(hidden_dim, num_movies)
        
        # 4. Knowledge Injection (Initialisierung)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder_l1.weight, a=math.sqrt(5))
            self.encoder_l1.weight[:num_genres, :].copy_(target_tensor)
            nn.init.zeros_(self.encoder_l2.weight)
            self.encoder_l2.weight[:, :num_genres].copy_(torch.eye(num_genres))

    def forward(self, x, user_overrides=None, alpha=1.0):
        """
        Bug 2 & 3 Fix: Parameter umbenannt von `user_category_dict` → `user_overrides`.
        Akzeptiert jetzt direkt einen Float-Tensor (1 x num_genres) statt eines String-Dicts,
        passend zum Aufruf in Schritt 7 des __main__-Blocks.
        
        Pfad A: KI-Autoencoder (immer aktiv)
        Pfad B: Hybrid Override (nur wenn user_overrides übergeben wird)
        """
        # Pfad A: KI-Autoencoder
        h_enc = self.encoder_act(self.encoder_l1(x))
        latent_profile = torch.sigmoid(self.encoder_l2(h_enc))
        
        h_dec = self.decoder_act(self.decoder_l1(self.dropout(latent_profile)))
        ann_output = torch.sigmoid(self.decoder_l2(h_dec)) * 5.0
        
        # Pfad B: Hybrid Override (user_overrides ist ein (1 x num_genres) Tensor)
        if user_overrides is not None:
            # explicit_bias: (1 x num_genres) @ (num_genres x num_movies) → (1 x num_movies)
            # Jedes Genre-Mitglied erhält sofort den mathematischen Push/Pull des Nutzers.
            explicit_bias = torch.matmul(user_overrides, self.target_genre_matrix)
            return ann_output + (alpha * explicit_bias), latent_profile
            
        return ann_output, latent_profile

    def apply_weight_clipping(self, epsilon=0.15):
        with torch.no_grad():
            # Encoder Layer 1 Clippen: erste num_genres Zeilen sollen nah an M_genre^T bleiben
            dev_l1 = self.encoder_l1.weight[:self.num_genres, :] - self.target_genre_matrix
            self.encoder_l1.weight[:self.num_genres, :].copy_(
                self.target_genre_matrix + torch.clamp(dev_l1, -epsilon, epsilon)
            )
            # Encoder Layer 2 Clippen: erste num_genres Spalten sollen nah an Identität bleiben
            identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
            dev_l2 = self.encoder_l2.weight[:, :self.num_genres] - identity
            self.encoder_l2.weight[:, :self.num_genres].copy_(
                identity + torch.clamp(dev_l2, -epsilon, epsilon)
            )

    def recommend_from_edited_profile(self, edited_profile: torch.Tensor) -> torch.Tensor:
        """
        HCAI Ebene 1: Der Mensch sitzt im Flaschenhals, nicht am Ausgang.
        
        Der Nutzer übergibt sein manuell editiertes Genre-Profil (1 x num_genres, Werte in [0..1]).
        Der Decoder übersetzt dieses menschliche Urteil direkt und vollständig in Filmscores —
        ohne KI-Korrekturfaktor, ohne Fusion. Reiner Human-to-Film-Pfad.
        
        Architektonischer Unterschied zum Hybrid Override Path:
          - forward() mit user_overrides:   KI schlägt vor → Mensch korrigiert den Output
          - recommend_from_edited_profile(): Mensch definiert das Profil → KI übersetzt in Scores
        
        Das ist konzeptuell der stärkste HCAI-Moment: Das Modell dient dem Menschen
        als Übersetzer, nicht als Entscheider.
        
        Args:
            edited_profile: Tensor (1 x num_genres), Genre-Affinitäten in [0, 1].
                            0.0 = kein Interesse, 1.0 = maximales Interesse.
        Returns:
            Filmscores (1 x num_movies) auf der Sterne-Skala [0..5].
        """
        h_dec = self.decoder_act(self.decoder_l1(edited_profile))
        return torch.sigmoid(self.decoder_l2(h_dec)) * 5.0

    def explain_override_impact(
        self,
        ann_output: torch.Tensor,
        final_output: torch.Tensor,
        idx_to_title: dict,
        top_n: int = 5
    ) -> dict:
        """
        HCAI Ebene 2: Transparenz über den Eingriff selbst.
        
        Zeigt dem Nutzer die konkrete Auswirkung seines Overrides:
        Welche Filme wurden am stärksten angehoben, welche abgewertet?
        Ohne diese Rückkopplung bleibt der Override eine Blackbox zweiter Ordnung —
        der Nutzer weiß, dass er steuert, aber nicht womit.
        
        Macht alpha und die Genre-Matrix für den Nutzer erlebbar: Die Ausgabe kann
        direkt im UI als "Dein Eingriff hat folgende Filme beeinflusst:" angezeigt werden.
        
        Args:
            ann_output:    KI-Output VOR dem Override  (1 x num_movies)
            final_output:  KI-Output NACH dem Override (1 x num_movies)
            idx_to_title:  Mapping dense_idx → Filmtitel
            top_n:         Anzahl der Top-Gewinner und -Verlierer
        
        Returns:
            Dict mit 'boosted' und 'suppressed' — je eine Liste von (Titel, Delta)-Tupeln.
        """
        delta = (final_output - ann_output).squeeze()           # (num_movies,)

        top_up_vals,   top_up_idx   = torch.topk(delta,  top_n)
        top_down_vals, top_down_idx = torch.topk(-delta, top_n)

        boosted = [
            (idx_to_title.get(i.item(), f"Film #{i.item()}"), v.item())
            for i, v in zip(top_up_idx, top_up_vals)
        ]
        suppressed = [
            (idx_to_title.get(i.item(), f"Film #{i.item()}"), -v.item())
            for i, v in zip(top_down_idx, top_down_vals)
        ]
        return {"boosted": boosted, "suppressed": suppressed}

    def get_semantic_loss(self):
        """
        Bug 1 Fix: Fehlende Methode ergänzt.
        Berechnet den quadratischen Drift der Encoder-Gewichte von ihrer semantischen Verankerung.
        
        Zweigliedriger Verlust:
          - L1-Anteil: encoder_l1.weight[:num_genres, :] soll M_genre^T (18x9742) widerspiegeln.
                       Diese Zeilen sind die direkten Genre-Detektoren des ersten Layers.
          - L2-Anteil: encoder_l2.weight[:, :num_genres] soll Identität (18x18) bleiben.
                       Diese Spalten leiten die Genre-Signale zum Flaschenhals durch.
        
        KRITISCH: torch.mean() statt torch.sum() verwenden, sonst dominiert der Semantic Loss
        numerisch den Vorhersage-Verlust (9742 * 18 = 175.356 Summanden) und blockiert das Lernen.
        """
        # Anteil 1: Drift der Genre-Detektor-Zeilen in encoder_l1
        loss_l1 = torch.mean(
            (self.encoder_l1.weight[:self.num_genres, :] - self.target_genre_matrix) ** 2
        )
        # Anteil 2: Drift der Durchleit-Spalten in encoder_l2 von der Identität
        identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
        loss_l2 = torch.mean(
            (self.encoder_l2.weight[:, :self.num_genres] - identity) ** 2
        )
        return loss_l1 + loss_l2

def generate_soft_xai_explanation(movie_title, movie_idx, model, single_user_profile):
    """
    Bug 4 Fix: Signatur von (movie_idx, model, latent_profile) auf
    (movie_title, movie_idx, model, single_user_profile) korrigiert —
    passend zum Aufruf in Schritt 8 des __main__-Blocks.
    
    Fuzzy Semantic Thresholding gemäß Spezifikation:
      - Kern-Genres:    eff_weight > 0.85  UND  Nutzer-Affinität > 0.60  → starke, explizite Nennung
      - Subtile Muster: 0.20 < eff_weight ≤ 0.85                         → ergänzende Transparenz
    
    Effektive Gewichte: encoder_l2.weight (18x128) @ encoder_l1.weight (128x9742)
    = (18x9742) — die vollständige Film→Genre-Projektionsmatrix der zusammengesetzten Encoderschichten.
    """
    with torch.no_grad():
        # Kompositions-Matrix: vollständige Projektion Film → Genre (18 x num_movies)
        eff_weights = model.encoder_l2.weight @ model.encoder_l1.weight
        weights = eff_weights[:, movie_idx].cpu().numpy()       # Gewichtsvektor für diesen Film
        profile = single_user_profile.cpu().numpy().flatten()   # Nutzer-Genre-Affinitäten [0..1]
    
    # Stufe 1: Kern-Genres (starke semantische Übereinstimmung + hohe Nutzer-Affinität)
    core_genres = [
        GENRES[i] for i in range(len(GENRES))
        if weights[i] > 0.85 and profile[i] > 0.6
    ]
    # Stufe 2: Subtile Muster (schwächere, aber noch relevante Genre-Signale)
    subtle_genres = [
        GENRES[i] for i in range(len(GENRES))
        if 0.20 < weights[i] <= 0.85 and profile[i] > 0.3
    ]
    
    if core_genres:
        explanation = (f'"{movie_title}" passt zu deinen starken Interessen in: '
                       f'{", ".join(core_genres)}.')
    else:
        explanation = (f'"{movie_title}" wird aufgrund allgemeiner kollaborativer '
                       f'Filtermuster empfohlen.')
    
    if subtle_genres:
        explanation += f' Subtile Genre-Einflüsse erkannt: {", ".join(subtle_genres)}.'
    
    return explanation
#endregion

#region loss & training utilities
def masked_mse_loss(predictions, targets, mask):
    """
    Custom Loss Function gegen das Sparsity-Problem.
    Berechnet den MSE ausschließlich auf den vom Nutzer tatsächlich bewerteten Filmen.
    """
    loss = (predictions - targets) ** 2
    masked_loss = loss * mask.float()
    num_elements = mask.float().sum()
    return masked_loss.sum() / (num_elements + 1e-8)

def train_step(model, optimizer, user_batch, mask_batch, lambda_reg=0.05, epsilon_clip=0.15):
    """
    Führt einen optimierten HCAI-Trainingsschritt aus.
    """
    optimizer.zero_grad()
    
    # HCAI Fix 1: Füttere den echten Bewertungsvektor ein, anstatt ihn fälschlicherweise komplett zu nullen.
    # Da wir einen Collaborative Autoencoder trainieren, füllen die Nullen ungesehene Filme implizit auf.
    # Die Eingliederung erfolgt sauber, weil `masked_mse_loss` die Verlustberechnung exakt auf echte Ratings beschränkt.
    predictions, _ = model(user_batch)
    
    # Kombinierte Verlustfunktion: Vorhersage-Güte + Elastische Semantik-Leine
    pred_loss = masked_mse_loss(predictions, user_batch, mask_batch)
    semantic_loss = model.get_semantic_loss()
    
    total_loss = pred_loss + (lambda_reg * semantic_loss)
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    # Sicherheitsnetz erzwingen: Gewichte zurückholen, falls sie zu weit driften
    model.apply_weight_clipping(epsilon=epsilon_clip)
    
    return total_loss.item(), pred_loss.item(), semantic_loss.item()
#endregion

#region explainable ai (XAI)

# ─────────────────────────────────────────────────────────────────────────────
# METHODE 1: SHAP (SHapley Additive exPlanations) — lokal, exakt
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_shap(
    model,
    user_profile:        torch.Tensor,
    movie_idx:           int,
    background_profiles: torch.Tensor,
    n_samples:           int = 100
) -> list | None:
    """
    KernelSHAP auf dem Decoder-Teilnetz für einen spezifischen Film.

    Warum nur der Decoder?
    Der Input — das latente Genre-Profil (18 Dim.) — ist in Ansatz 1 bereits
    menschlich interpretierbar. SHAP-Werte hier beantworten direkt:
    "Wie viele Sterne hat mein Sci-Fi-Interesse zu diesem Film beigetragen?"

    Paper-Argument: In Ansatz 0 sind dieselben SHAP-Werte über willkürliche,
    unbenannte Bottleneck-Dimensionen verteilt — mathematisch berechenbar,
    semantisch wertlos. In Ansatz 1 tragen die Werte direkt Genrenamen.

    Dimensionen:
        background_profiles: (k, 18)  — Hintergrunddaten für den Erwartungswert
        user_profile:        (1, 18)  — zu erklärendes Nutzerprofil
        → SHAP-Werte:        (18,)    — Beitrag jedes Genres in Sterne-Einheiten

    Args:
        model:               Trainiertes SoftRegularizedHCAIAutoEncoder-Modell
        user_profile:        Genre-Profil des Nutzers (1 x num_genres)
        movie_idx:           Dichter Tensor-Index des zu erklärenden Films
        background_profiles: Sample von Nutzerprofilen als SHAP-Baseline (k x num_genres)
        n_samples:           SHAP-Perturbationsanzahl (mehr = genauer, aber langsamer)

    Returns:
        Liste von (Genre, SHAP-Wert) Tupeln, absteigend nach absolutem Einfluss sortiert.
        None wenn shap nicht installiert ist.
    """
    if not SHAP_AVAILABLE:
        print("   ⚠️  shap nicht installiert — übersprungen. (pip install shap)")
        return None

    # Decoder-Scoring-Funktion: (n, 18) np.array → (n,) Scores für movie_idx
    # Verwendet recommend_from_edited_profile-Logik ohne Dropout (eval-Modus)
    def decoder_score_fn(profiles_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(profiles_np, dtype=torch.float32)
            h = model.decoder_act(model.decoder_l1(t))
            scores = torch.sigmoid(model.decoder_l2(h)) * 5.0
            return scores[:, movie_idx].numpy()

    background_np = background_profiles.numpy()                  # (k, 18)
    explainer     = shap.KernelExplainer(decoder_score_fn, background_np)

    profile_np  = user_profile.numpy().reshape(1, -1)            # (1, 18)
    shap_values = explainer.shap_values(                         # (1, 18) → flatten → (18,)
        profile_np, nsamples=n_samples, silent=True
    )

    contributions = list(zip(GENRES, shap_values.flatten()))
    return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# METHODE 2: LIME (Local Interpretable Model-agnostic Explanations) — lokal, approximativ
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(
    model,
    user_profile: torch.Tensor,
    movie_idx:    int,
    n_samples:    int = 300
) -> list | None:
    """
    LIME im latenten Genre-Raum statt im rohen Film-Raum (9742 Dim.).

    Warum Genre-Raum statt Film-Raum?
    Naives LIME auf dem 9742-dimensionalen Input wäre extrem teuer und
    liefert uninterpretierbare Feature-Wichtigkeiten ("Film #4823 beeinflusst
    Film #1337"). LIME auf den 18 Genre-Neuronen ist schnell, stabil und
    direkt lesbar.

    Methodik:
        1. Erzeuge n_samples leicht verrauschte Varianten des Nutzerprofils
           im Genre-Raum (Gauß-Noise, σ=0.15, geclampt auf [0,1]).
        2. Berechne für jede Variante den Decoder-Score für movie_idx.
        3. Fitte ein lokales lineares Modell (Ridge Regression).
        4. Koeffizienten = lokale Genre-Wichtigkeiten für diesen Nutzer/Film.

    Limitation (Paper-relevant):
        LIME ist instabil — zwei Läufe mit verschiedenem Seed können abweichende
        Koeffizienten liefern. Für quantitative Paper-Tabellen SHAP bevorzugen;
        LIME eignet sich als Kontrast-Demonstration der Instabilität in Ansatz 0.

    Args:
        model:        Trainiertes Modell (eval-Modus empfohlen)
        user_profile: Genre-Profil des Nutzers (1 x num_genres)
        movie_idx:    Dichter Tensor-Index des zu erklärenden Films
        n_samples:    Anzahl der Perturbationssamples (300 ist für 18 Features ausreichend)

    Returns:
        Liste von (Genre, Koeffizient) Tupeln, absteigend nach absolutem Einfluss sortiert.
        None wenn scikit-learn nicht installiert ist.
    """
    if not SKLEARN_AVAILABLE:
        print("   ⚠️  scikit-learn nicht installiert — übersprungen. (pip install scikit-learn)")
        return None

    profile_np = user_profile.numpy().flatten()                   # (18,)

    # Perturbierte Genre-Profile erzeugen (lokale Nachbarschaft)
    np.random.seed(42)
    noise     = np.random.normal(0, 0.15, size=(n_samples, len(GENRES)))
    perturbed = np.clip(profile_np + noise, 0.0, 1.0).astype(np.float32)  # (n, 18)

    # Decoder-Scores für alle perturbierten Profile in einem einzigen Batch
    with torch.no_grad():
        t      = torch.tensor(perturbed, dtype=torch.float32)
        h      = model.decoder_act(model.decoder_l1(t))
        scores = (torch.sigmoid(model.decoder_l2(h)) * 5.0)[:, movie_idx].numpy()  # (n,)

    # Lokales lineares Modell: Koeffizient[i] = Einfluss von Genre i auf den Score
    local_model = Ridge(alpha=1.0)
    local_model.fit(perturbed, scores)

    contributions = list(zip(GENRES, local_model.coef_))
    return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# METHODE 3: Permutation Feature Importance — global, modell-agnostisch
# ─────────────────────────────────────────────────────────────────────────────

def _eval_decoder_mse(
    model,
    profiles: torch.Tensor,
    targets:  torch.Tensor
) -> float:
    """
    Interne Hilfsfunktion: Berechnet den masked MSE des Decoders über alle Nutzer.
    Verwendet recommend_from_edited_profile-Logik (kein Dropout, kein Encoder-Pfad).
    """
    with torch.no_grad():
        h     = model.decoder_act(model.decoder_l1(profiles))
        preds = torch.sigmoid(model.decoder_l2(h)) * 5.0
        mask  = targets > 0.0
        return masked_mse_loss(preds, targets, mask).item()


def compute_genre_permutation_importance(
    model,
    all_profiles: torch.Tensor,
    all_targets:  torch.Tensor,
    n_repeats:    int = 5
) -> dict:
    """
    Globale Permutation Feature Importance über alle Nutzerprofile.

    Für jedes der 18 Genre-Neuronen: Wie stark steigt der mittlere masked MSE
    über alle 610 Nutzer, wenn diese Dimension zufällig permutiert wird?

    Hohe Wichtigkeit → Dieses Genre ist global entscheidend für die Qualität
    aller Empfehlungen. Niedriger Wert → Das Genre trägt kaum zum Signal bei
    (aber es ist im Datensatz vielleicht ohnehin selten vertreten).

    Paper-Argument (Kernaussage):
        Ansatz 0: Die wichtigsten Bottleneck-Dimensionen sind unbenannte Zahlen.
                  Der Reviewer kann sie nicht interpretieren.
        Ansatz 1: Die wichtigsten Dimensionen korrespondieren mit echten Genres
                  (in MovieLens empirisch: Drama, Comedy, Thriller dominieren).
                  Das macht das Modellverhalten global erklärbar und prüfbar.

    Args:
        model:        Trainiertes Modell (eval-Modus)
        all_profiles: Latente Genre-Profile aller Nutzer (num_users x num_genres)
        all_targets:  Echte Bewertungsmatrix R (num_users x num_movies), 0 = unbewertet
        n_repeats:    Anzahl der Permutationswiederholungen pro Genre (Varianzreduktion)

    Returns:
        Dict {Genre: mittlerer MSE-Anstieg}, absteigend nach Wichtigkeit sortiert.
    """
    baseline = _eval_decoder_mse(model, all_profiles, all_targets)
    importances = {}

    for g_idx, genre in enumerate(GENRES):
        deltas = []
        for _ in range(n_repeats):
            permuted = all_profiles.clone()
            # Permutiere nur die eine Genre-Spalte — alle anderen bleiben unverändert
            perm_order = torch.randperm(permuted.size(0))
            permuted[:, g_idx] = permuted[perm_order, g_idx]
            deltas.append(_eval_decoder_mse(model, permuted, all_targets) - baseline)
        importances[genre] = float(np.mean(deltas))

    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

#endregion

#region comprehensive real dataset testing

if __name__ == "__main__":
    import torch.optim as optim
    import time
    
    print("=" * 70)
    print("🎬 STARTE COMPREHENSIVE MOVIELENS DATASET EVALUATION PIPELINE")
    print("=" * 70)
    
    # PRÜFUNG: Existiert der Datensatz-Ordner?
    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, 'ratings.csv')):
        print(f"❌ FEHLER: Der Ordner '{DATA_DIR}' oder die CSV-Dateien wurden nicht gefunden!")
        print("Bitte lade den 'ml-latest-small'-Datensatz von MovieLens herunter und entpacke ihn in dieses Verzeichnis.")
        exit()

    print("⏳ Schritt 1: Lade originale MovieLens-Dateien...")
    ratings_df, movies_df = load_movielens_data()
    print(f"-> {len(ratings_df)} Bewertungen und {len(movies_df)} Filme erfolgreich geladen.")

    print("\n⚙️ Schritt 2: Bereite Daten-Alignment vor (Mapping lückenhafter Movie-IDs)...")
    # Mapping-Dictionairies für dichte PyTorch-Indizes erstellen
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'].unique())}
    
    # HCAI Fix 3a: Korrekter Aufbau des Titel-Wörterbuchs über die dichten Indizes.
    # Löst die Diskrepanz zwischen fehlerhaften Pandas-Zeilennummern und echten Tensor-IDs.
    idx_to_movie_title = {}
    for _, row in movies_df.iterrows():
        m_id = int(row['movieId'])
        if m_id in movie_id_to_idx:
            dense_idx = movie_id_to_idx[m_id]
            idx_to_movie_title[dense_idx] = row['title']
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
    
    num_users = len(user_id_to_idx)
    num_movies = len(movie_id_to_idx)
    num_genres = len(GENRES)
    
    print(f"-> Dimensionen der dichten Matrix: {num_users} Nutzer x {num_movies} Filme")

    print("\n📊 Schritt 3: Baue dichte User-Movie-Bewertungsmatrix und Ur-Genre-Matrix auf...")
    # 3.1 Bewertungsmatrix (R)
    R_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    for _, row in ratings_df.iterrows():
        u_idx = user_id_to_idx[int(row['userId'])]
        m_idx = movie_id_to_idx[int(row['movieId'])]
        R_matrix[u_idx, m_idx] = float(row['rating'])
        
    # 3.2 Alignierte Genre-Matrix (M_genre)
    pure_genre_matrix_np = np.zeros((num_movies, num_genres), dtype=np.float32)
    for _, row in movies_df.iterrows():
        m_idx = movie_id_to_idx[int(row['movieId'])]
        movie_genres = row['genres'].split('|')
        for g in movie_genres:
            if g in GENRES:
                pure_genre_matrix_np[m_idx, GENRES.index(g)] = 1.0

    print("-> Matrizen erfolgreich im RAM initialisiert.")
    sparsity = (1.0 - (np.count_nonzero(R_matrix) / R_matrix.size)) * 100
    print(f"-> Datensatz-Sparsity: {sparsity:.2f}% (Extrem dünn besetzt!)")

    print("\n🧠 Schritt 4: Initialisiere Soft-Regularized HCAI Autoencoder...")
    model = SoftRegularizedHCAIAutoEncoder(num_movies, num_genres, pure_genre_matrix_np)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    print(f"-> Gewichte des Encoders mit {pure_genre_matrix_np.shape}-Genre-Fakten vorinitialisiert.")

    print("\n🏋️‍♂️ Schritt 5: Starte Modelltraining über alle 610 echten Nutzer...")
    R_tensor = torch.tensor(R_matrix, dtype=torch.float32)
    global_evaluation_mask = R_tensor > 0.0
    
    EPOCHS = 25
    BATCH_SIZE = 32
    
    start_time = time.time()
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_semantic_loss = 0.0
        batches = 0
        
        permutation = torch.randperm(R_tensor.size(0))
        for i in range(0, R_tensor.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_users = R_tensor[indices]
            batch_mask = global_evaluation_mask[indices]
            
            t_loss, p_loss, s_loss = train_step(
                model, optimizer, batch_users, batch_mask, lambda_reg=0.05, epsilon_clip=0.15
            )
            
            epoch_loss += t_loss
            epoch_pred_loss += p_loss
            epoch_semantic_loss += s_loss
            batches += 1
            
        print(f"🔹 Epoche {epoch+1}/{EPOCHS} | Total Loss: {epoch_loss/batches:.4f} " 
              f"| Pred Loss (MSE): {epoch_pred_loss/batches:.4f} | Semantic Loss (Drift): {epoch_semantic_loss/batches:.4f}")

    print(f"-> Training abgeschlossen in {time.time() - start_time:.2f} Sekunden.")

    print("\n🔬 Schritt 6: Evaluiere das System an einem konkreten echten Nutzer...")
    model.eval()
    test_user_idx = 1 
    
    single_user_vector = R_tensor[test_user_idx].unsqueeze(0)
    
    with torch.no_grad():
        ki_predictions, latent_profile = model(single_user_vector)
    
    print(f"-> Nutzer-Index {test_user_idx} geladen.")
    print("\n📊 Von der KI extrahiertes, implizites Genre-Interessenprofil (Top 5 Genres):")
    profile_np = latent_profile[0].numpy()
    sorted_genres_idx = np.argsort(profile_np)[::-1]
    for idx in sorted_genres_idx[:5]:
        print(f"   * {GENRES[idx]}: {profile_np[idx]*100:.1f}% Match-Wahrscheinlichkeit")

    print("\n🎛️ Schritt 7: Simuliere HCAI User Intervention (Nutzer-Eingriff im UI)...")
    # HCAI Ebene 3: Nutzer drückt sich in Sprache aus — build_override_tensor übersetzt in Mathematik.
    # OVERRIDE_MAP ist jetzt die Single Source of Truth; keine Magic-Numbers mehr im Testcode.
    user_ui_overrides = build_override_tensor({
        "Sci-Fi":  "stark verstärken",   # +2.0
        "Comedy":  "stark dämpfen",      # -2.0
    }, num_genres)
    
    with torch.no_grad():
        hybrid_predictions, _ = model(single_user_vector, user_overrides=user_ui_overrides, alpha=3.0)
    
    print("\n📈 Auswirkung des UI-Eingriffs auf reale Filmbewertungen (Scores):")
    
    # HCAI Fix 3b: Suchen der exakten dichten PyTorch-Indizes über das reale ID-Mapping.
    # Schützt das System vor Falschzuordnungen und Index-Errors während der Laufzeit.
    matrix_movie_id = movies_df[movies_df['title'].str.contains("Matrix, The", na=False)]['movieId'].values[0]
    matrix_movie_idx = movie_id_to_idx[matrix_movie_id]
    
    hangover_movie_id = movies_df[movies_df['title'].str.contains("Hangover, The", na=False)]['movieId'].values[0]
    hangover_movie_idx = movie_id_to_idx[hangover_movie_id]
    
    print(f"🎥 {idx_to_movie_title[matrix_movie_idx]} (Sci-Fi):")
    print(f"   * Score OHNE UI-Regler: {ki_predictions[0, matrix_movie_idx]:.4f}")
    print(f"   * Score MIT Sci-Fi (+2.5) Regler: {hybrid_predictions[0, matrix_movie_idx]:.4f} 🚀")
    
    print(f"🎥 {idx_to_movie_title[hangover_movie_idx]} (Comedy):")
    print(f"   * Score OHNE UI-Regler: {ki_predictions[0, hangover_movie_idx]:.4f}")
    print(f"   * Score MIT Comedy (-2.5) Regler: {hybrid_predictions[0, hangover_movie_idx]:.4f} 📉")

    print("\n📝 Schritt 8: Teste die Explainable AI (XAI) Engine auf echten Daten...")
    xai_explanation = generate_soft_xai_explanation(
        movie_title=idx_to_movie_title[matrix_movie_idx],
        movie_idx=matrix_movie_idx,
        model=model,
        single_user_profile=latent_profile[0]
    )
    print(f"Generierte Transparenz-Erklärung für den Nutzer:")
    print(f"-> \"{xai_explanation}\"")

    print("\n🧬 Schritt 9: Teste Direkte Profil-Editierung (HCAI Ebene 1)...")
    # Der Nutzer definiert sein ideales Genre-Profil manuell — der Decoder übersetzt es in Filme.
    # Das ist der konzeptuell stärkste HCAI-Moment: Kein KI-Output, kein Fusionsfaktor.
    # Der Mensch sitzt im Flaschenhals. Das Modell dient als reiner Übersetzer.
    edited_profile = latent_profile.clone()  # Start: KI-berechnetes Profil als Ausgangspunkt
    edited_profile[0, GENRES.index("Sci-Fi")]   = 0.95  # Nutzer setzt Sci-Fi auf Maximum
    edited_profile[0, GENRES.index("Comedy")]   = 0.05  # Nutzer unterdrückt Comedy fast vollständig
    edited_profile[0, GENRES.index("Thriller")] = 0.80  # Nutzer verstärkt Thriller stark
    
    with torch.no_grad():
        profile_driven_output = model.recommend_from_edited_profile(edited_profile)
    
    print("-> Nutzer hat Profil direkt editiert: Sci-Fi=95%, Comedy=5%, Thriller=80%")
    print("-> Top 5 Empfehlungen aus menschlich definiertem Profil:")
    top5_profile_driven = torch.topk(profile_driven_output.squeeze(), 5)
    for rank, (score, idx) in enumerate(zip(top5_profile_driven.values, top5_profile_driven.indices), 1):
        title = idx_to_movie_title.get(idx.item(), f"Film #{idx.item()}")
        print(f"   {rank}. {title} — Score: {score.item():.4f}")

    print("\n🔍 Schritt 10: Erkläre den Eingriff (HCAI Ebene 2 — Delta-Transparenz)...")
    # Zeigt dem Nutzer, was sein Override in Schritt 7 konkret bewirkt hat.
    # Macht alpha und die Genre-Matrix erlebbar, statt sie als Blackbox zu lassen.
    with torch.no_grad():
        impact = model.explain_override_impact(
            ann_output=ki_predictions,
            final_output=hybrid_predictions,
            idx_to_title=idx_to_movie_title,
            top_n=5
        )
    
    print("-> Filme, die durch deinen Sci-Fi-Boost am stärksten gestiegen sind:")
    for title, delta in impact["boosted"]:
        print(f"   ▲ {title}: +{delta:.4f} Punkte")
    
    print("-> Filme, die durch deinen Comedy-Dämpfer am stärksten gefallen sind:")
    for title, delta in impact["suppressed"]:
        print(f"   ▼ {title}: -{delta:.4f} Punkte")

    # ─────────────────────────────────────────────────────────────────────────
    # SCHRITTE 11–14: XAI-Methoden (SHAP / LIME / Permutation Importance)
    # ─────────────────────────────────────────────────────────────────────────

    print("\n🧮 Schritt 11: Berechne latente Profile aller Nutzer (XAI-Vorbereitung)...")
    # Einmaliger Encoder-Durchlauf über alle 610 Nutzer.
    # all_latent_profiles dient als: (a) SHAP-Hintergrunddaten, (b) Input für Perm. Importance.
    model.eval()
    with torch.no_grad():
        _, all_latent_profiles = model(R_tensor)   # (610, 18)
    print(f"-> {all_latent_profiles.shape[0]} Nutzerprofile berechnet "
          f"(Shape: {list(all_latent_profiles.shape)}).")

    print(f"\n🔷 Schritt 12: SHAP-Analyse für '{idx_to_movie_title[matrix_movie_idx]}'...")
    print("   (KernelSHAP auf dem Decoder — kann 10–30 Sekunden dauern)")
    # Hintergrund: 30 zufällige Nutzerprofile als Baseline für den Shapley-Erwartungswert.
    # Mehr Hintergrundprofile = stabilere Werte, aber längere Laufzeit.
    bg_indices      = torch.randperm(all_latent_profiles.size(0))[:30]
    background_prfs = all_latent_profiles[bg_indices]

    shap_result = explain_with_shap(
        model            = model,
        user_profile     = latent_profile[0],
        movie_idx        = matrix_movie_idx,
        background_profiles = background_prfs,
        n_samples        = 100
    )
    if shap_result:
        print(f"   SHAP-Werte (Genre-Beitrag in Sternen zu diesem Film):")
        for genre, val in shap_result[:6]:
            bar   = "▲" if val > 0 else "▼"
            print(f"   {bar} {genre:<15} {val:+.4f} ⭐")

    print(f"\n🟡 Schritt 13: LIME-Analyse für '{idx_to_movie_title[matrix_movie_idx]}'...")
    lime_result = explain_with_lime(
        model        = model,
        user_profile = latent_profile[0],
        movie_idx    = matrix_movie_idx,
        n_samples    = 300
    )
    if lime_result:
        print(f"   LIME-Koeffizienten (lokale Genre-Wichtigkeit, seed=42):")
        for genre, coef in lime_result[:6]:
            bar = "▲" if coef > 0 else "▼"
            print(f"   {bar} {genre:<15} {coef:+.4f}")
        print("   ℹ️  LIME-Tipp: Bei erneutem Aufruf mit anderem Seed können "
              "Werte abweichen — das ist ein bekanntes Stabilitätsproblem von LIME.")

    print("\n🌍 Schritt 14: Globale Permutation Feature Importance (alle 610 Nutzer)...")
    perm_importance = compute_genre_permutation_importance(
        model         = model,
        all_profiles  = all_latent_profiles,
        all_targets   = R_tensor,
        n_repeats     = 5
    )
    print("   Globale Genre-Wichtigkeit (MSE-Anstieg bei Permutation):")
    for rank, (genre, delta) in enumerate(perm_importance.items(), 1):
        bar = "█" * max(1, int(delta * 200))
        print(f"   {rank:>2}. {genre:<15} Δ MSE: {delta:+.5f}  {bar}")
    top_genre = next(iter(perm_importance))
    print(f"\n   → Das global wichtigste Genre für Ansatz 1: '{top_genre}'")
    print("   → In Ansatz 0 wäre diese Dimension eine unbenannte Zahl.")

    print("=" * 70)

#endregion