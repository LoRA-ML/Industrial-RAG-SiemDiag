"""
SiemDiag : assistant de diagnostic RAG pour les automates Siemens S7-1500 et variateurs SINAMICS G120C.

Lancement (depuis la racine du projet, environnement virtuel activé) :
    python app.py                 # interface sur http://127.0.0.1:7860

Le script charge successivement le corpus, les modèles d'embedding et de re-ranking, l'index de recherche,
le LLM Qwen 2.5 3B Instruct (quantifié 4 bits sur GPU) et son adaptateur LoRA, puis ouvre l'interface Gradio.
Au premier lancement, les modèles sont téléchargés depuis Hugging Face (~10 Go au total).
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RESOURCES = ROOT / "ressources"

LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
EMB_ID = "intfloat/multilingual-e5-base"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"

REFUSAL = "Information absente de la documentation fournie."


# ======================================================================================
# Affichage terminal
# ======================================================================================

_T0 = time.time()


def step(n, total, title):
    print(f"\n[{n}/{total}] {title}", flush=True)


def info(msg):
    print(f"      {msg}", flush=True)


def elapsed():
    return f"{time.time() - _T0:.0f} s"


# ======================================================================================
# Agent conversationnel (historique de taille contrôlée, prompt système, génération)
# ======================================================================================

class BaselineAgent:

    def __init__(self, tokenizer, model, max_history=3):
        self.tokenizer = tokenizer
        self.model = model
        # Contrôle de la mémoire tampon (Taille maximale de l'historique de dialogue)
        self.max_history = max_history
        self.history = []
        self.few_shot = []  # exemples permanents, jamais tronqués
        self.system_prompt = ""

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def add_example_few_shot(self, user_text, agent_text):
        """
          Permet d'injecter des exemples de comportement (few-shot)
        """
        self.few_shot.append({"role": "user", "content": user_text})
        self.few_shot.append({"role": "assistant", "content": agent_text})

    def reset_chat(self):
        self.history = []

    def reset_examples_few_shot(self):
        self.few_shot = []

    def _build_generation(self, user_msg, sent_msg, max_new_tokens, temp, top_k, top_p, seed=None):
        # Limitation de la taille de l'historique pour éviter l'explosion de contexte
        # Chaque tour complet comprend 1 'user' et 1 'agent' donc max_history * 2
        context_window = self.history[-(self.max_history * 2):]

        messages = []
        if self.system_prompt:  # si system prompt pas vide, l'ajouter au contexte en tant que premier message
            messages.append({"role": "system", "content": self.system_prompt})

        # Ajouter au contexte les exemples few shot, la fenêtre actuelle de contexte et le message envoyé
        messages += self.few_shot + context_window + [{"role": "user", "content": sent_msg}]

        # Application du format de conversation attendu nativement par le tokenizer (ChatML ici)
        prompt_format = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenisation
        inputs = self.tokenizer(prompt_format, return_tensors="pt").to(self.model.device)

        if seed is not None:  # pour la reproducibilité du tirage
            torch.manual_seed(seed)

        doSample = temp > 0
        # Génération avec paramètres de décodage
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # évite les réponses inutilement verbeuses et hors-sujet
                do_sample=doSample,  # sinon argmax par défaut
                temperature=temp if doSample else None,  # éviter division par 0 ou négatif
                top_p=top_p if doSample else None,
                top_k=top_k if doSample else None,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Extraction et nettoyage de la réponse générée
        input_length = inputs.input_ids.shape[1]
        raw_answer = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        # Enregistrement de la réponse de l'assistant dans le véritable historique
        # On n'enregistre le tour qu'une fois la génération réussie (pas de message orphelin)
        self.history += [{"role": "user", "content": user_msg},  # on enregistre dans l'historique que la question/message envoyé, pas les docs envoyés avec
                         {"role": "assistant", "content": raw_answer}]

        return raw_answer.strip()

    def generate(self, user_msg, context=None, max_new_tokens=300, temp=0.7, top_k=20, top_p=0.8, seed=None):
        """
          Méthode d'interface pour générer une réponse
        """
        sent_msg = user_msg
        if context:  # contexte ajouté qu'au message envoyé, pas à l'historique
            sent_msg = f"Extraits de documentation :\n{context}\n\nQuestion : {user_msg}"

        return self._build_generation(user_msg, sent_msg, max_new_tokens, temp, top_k, top_p, seed)


# ======================================================================================
# Reformulation et traduction de la question
# ======================================================================================

CONDENSE_PROMPT = "Réécris la dernière question pour qu'elle soit compréhensible sans l'historique. Ne réponds pas, renvoie uniquement la question réécrite."

TRANSLATE_PROMPT = ("Traduis la question en anglais technique (automatismes Siemens). "
                    "Conserve tels quels les codes, numéros et noms de produits. "
                    "Renvoie uniquement la traduction.")


def _short_generation(agent, system, user, max_new_tokens):
    """
    Génération courte et déterministe (argmax), utilisée pour la reformulation et la traduction.
    """
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    p = agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # appliquer le template du modèle
    ids = agent.tokenizer(p, return_tensors="pt").to(agent.model.device)  # tokeniser le prompt et envoie sur le GPU du modèle
    with torch.no_grad():
        out = agent.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return agent.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()  # découpe la réponse générée du prompt et décode en texte


def condense(agent, history, question):
    """
    Reformule une question de l'utilisateur pour qu'elle soit compréhensible sans le reste de l'historique de conversation.
    Exemple : gère les questions "Et pour [...] ?" à la suite d'une réponse à une même question.
    Remonte à maximum 4 messages (pour éviter de gâcher trop de tokens, en général les questions liées sont proches).
    """
    if not history:  # première question, pas d'historique
        return question

    # Garder les 4 derniers message de l'historique (au cas où rapport avec la question actuelle)
    # Formattage "role: content"
    hist = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-4:])
    return _short_generation(agent, CONDENSE_PROMPT, f"Historique :\n{hist}\n\nQuestion : {question}", max_new_tokens=60)


def translate_en(agent, question):
    """
    Traduit la question en anglais pour élargir la recherche aux passages des manuels anglais.
    Argmax pour une traduction déterministe, comme dans condense().
    """
    return _short_generation(agent, TRANSLATE_PROMPT, question, max_new_tokens=80)


# ======================================================================================
# Corpus Siemens
# ======================================================================================

def load_jsonl(path):
    """
    Ouvrir un fichier jsonl et le désérialiser.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def page_range(c):
    """
    Retourne le format de référence d'une page ou d'un intervalle de pages.
    Exemples : "p.12", "p.12-15"
    """
    return f"p.{c['page_debut']}" if c["page_debut"] == c["page_fin"] else f"p.{c['page_debut']}-{c['page_fin']}"


def to_pipeline_chunk(c, corpus_dir):
    """
    Convertit un passage de chunks.jsonl au format utilisé par la pipeline.
    """
    edition = f", éd. {c['edition']}" if c.get("edition") else ""
    return {
        "id": c["chunk_id"],          # identifiant stable, ex. d13_c01947
        "doc_id": c["doc_id"],        # numéro du manuel (1 à 21) : sert à la diversité et au filtrage
        "code": c.get("code") or "",  # code défaut/paramètre pour les entrées (F07801, p3112...), vide sinon
        "lang": c["lang"],            # FR ou EN : un passage anglais peut répondre à une question française
        "type": c["type"],            # entry / table / text / figure
        "page_debut": c["page_debut"],
        "page_fin": c["page_fin"],
        # dernier niveau du chemin de chapitre : les niveaux supérieurs contiennent parfois des artefacts d'extraction
        "section": (c.get("section") or "").split(" > ")[-1],
        "ref": f"{c['doc']}{edition}, {page_range(c)}",  # référence complète à citer
        "text": c["text"],            # texte transmis au LLM
        "embed": c["text_embed"],     # texte + manuel + chapitre : vectorisé, indexé par BM25, lu par le reranker
        "images": [str(corpus_dir / p) for p in c.get("images") or []],  # schémas rattachés pour les montrer
    }


# ======================================================================================
# Index de recherche hybride : FAISS + BM25 + codes exacts, re-ranking, diversité
# ======================================================================================

# Codes qu'un technicien peut taper : défauts/alarmes du variateur (F30001, A01028),
# paramètres (r0947, p1082), codes d'erreur des modules d'E/S (0x0006)
CODE_RE = re.compile(r"\b(?:[FACN]\d{5}|[rp]\d{4}|0x[0-9A-F]{3,4})\b", re.IGNORECASE)


def find_codes(text):
    """
    Détection des codes présents dans la question, remis à la casse du corpus.
    Exemples : f07801 -> F07801, P3112 -> p3112.
    """
    out = []
    for m in CODE_RE.findall(text):
        if m[:2].lower() == "0x":
            out.append("0x" + m[2:].upper())
        elif m[0] in "rpRP":
            out.append(m.lower())
        else:
            out.append(m.upper())
    return out


def strip_accents(t):
    """
    'défaut' -> 'defaut' : BM25 doit retrouver le même mot, accentué ou non.
    """
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def tok_bm25(t):
    """
    Tokenisation de t pour BM25.
    """
    return re.findall(r"\w+", strip_accents(t).lower())


class SiemensIndex:
    def __init__(self, chunks, emb_model, reranker, cache_path=None):
        self.chunks = chunks
        self.emb_model = emb_model
        self.reranker = reranker

        # Encodage des chunks (passages) en vecteurs denses d'embeddings pour E5-base
        # Préfixe "passage:" obligatoire pour E5 (sans lui, la recherche se dégrade sans erreur visible).
        # L'encodage des passages ne se fait qu'une fois : les vecteurs sont mis en cache sur disque.
        if cache_path and os.path.exists(cache_path) and len(np.load(cache_path, mmap_mode="r")) == len(chunks):
            E = np.load(cache_path)
            info(f"Vecteurs chargés depuis le cache {cache_path}")
        else:
            info("Cache de vecteurs absent ou obsolète : encodage de tous les passages (plusieurs minutes)...")
            t0 = time.time()
            E = emb_model.encode([f"passage: {c['embed']}" for c in chunks],
                                 batch_size=64,
                                 normalize_embeddings=True,
                                 show_progress_bar=True).astype("float32")
            info(f"Encodage en {(time.time() - t0) / 60:.1f} min")
            if cache_path:
                np.save(cache_path, E)
                info(f"Vecteurs sauvegardés dans {cache_path}")

        # index FAISS auquel on ajoute tous les vecteurs
        self.faiss = faiss.IndexFlatIP(E.shape[1])
        self.faiss.add(E)
        info(f"Index FAISS : {self.faiss.ntotal} vecteurs de dimension {E.shape[1]}")

        # Index lexical BM25, sur le texte enrichi du manuel et du chapitre, sans accents
        self.bm25 = BM25Okapi([tok_bm25(c["embed"]) for c in chunks])
        info("Index BM25 construit")

        # Table code -> passages, pour la correspondance exacte sur les entrées de catalogue
        self.by_code = {}
        for i, c in enumerate(chunks):
            if c["code"]:
                self.by_code.setdefault(c["code"], []).append(i)
        info(f"Table des codes : {len(self.by_code)} codes défaut/alarme/paramètre")

    def _candidates(self, query, k, docs=None, rrf_k=60):
        """
        Classement hybride FAISS + BM25 fusionné par RRF, comme dans HybridIndex.
        Renvoie : {indice_passage: score_RRF}.
        """

        # Encode la requête en embeddings avec E5-base
        q = self.emb_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")

        # avec un filtre de manuels, on classe tout le corpus puis on ne garde que les manuels demandés
        n = self.faiss.ntotal if docs else k
        _, dense_ids = self.faiss.search(q, n)
        bm_ids = np.argsort(self.bm25.get_scores(tok_bm25(query)))[::-1]

        # Reciprocal Rank Fusion : combine les 2 classements (FAISS, BM25)
        scores = {}
        for ranking in (dense_ids[0], bm_ids):
            if docs:
                ranking = [i for i in ranking if self.chunks[i]["doc_id"] in docs]
            for r, i in enumerate(ranking[:k]):
                scores[int(i)] = scores.get(int(i), 0) + 1 / (rrf_k + r + 1)
        return scores

    def search(self, query, extra_queries=(), k=50, n_rerank=30, final_k=5, max_per_doc=2, docs=None, max_figures=1):
        """
        Renvoie les final_k meilleurs passages pour la question, sous forme [(score reranker, chunk)].
        - extra_queries : reformulations dont les candidats rejoignent le même pool (ex. traduction anglaise)
        - max_per_doc : au plus N passages par manuel (0 = pas de limite)
        - docs : restreindre à certains manuels, ex. docs=[13, 19] pour les List Manuals du variateur
        - max_figures : au plus N passages de type figure en tête (None = pas de limite)
        """
        # Pool de candidats : les scores RRF de chaque formulation de la question s'additionnent
        pool = {}
        for q in (query, *extra_queries):
            for i, s in self._candidates(q, k, docs).items():
                pool[i] = pool.get(i, 0) + s
        top = sorted(pool, key=pool.get, reverse=True)[:n_rerank]

        # Entrées de catalogue portant exactement un code cité dans la question
        exact = [i for code in find_codes(query) for i in self.by_code.get(code, [])
                 if not docs or self.chunks[i]["doc_id"] in docs]
        candidates = list(dict.fromkeys(exact + top))  # dédoublonne en conservant l'ordre

        # Re-ranking cross-encoder sur (question d'origine, texte enrichi), score entre 0 et 1
        rr = self.reranker.predict([(query, self.chunks[i]["embed"]) for i in candidates])
        ranked = sorted(zip(rr, candidates), key=lambda x: -x[0])

        # Codes exacts en tête, puis diversité : au plus max_per_doc passages par manuel,
        # complété par les passages écartés s'il n'y a pas assez de manuels différents
        exact = set(exact)
        ranked = [x for x in ranked if x[1] in exact] + [x for x in ranked if x[1] not in exact]
        kept, rest, per_doc = [], [], Counter()

        for s, i in ranked:
            d = self.chunks[i]["doc_id"]
            if i in exact or not max_per_doc or per_doc[d] < max_per_doc:
                kept.append((s, i))
                per_doc[d] += 1
            else:
                rest.append((s, i))
        ranked = kept + rest

        # Au plus max_figures passages "figure" (libellés + texte OCR, bruité) en tête, les suivants passent derrière
        if max_figures is not None:
            figs = [x for x in ranked if self.chunks[x[1]]["type"] == "figure" and x[1] not in exact]
            extra_figs = {i for _, i in figs[max_figures:]}
            ranked = [x for x in ranked if x[1] not in extra_figs] + figs[max_figures:]

        return [(float(s), self.chunks[i]) for s, i in ranked[:final_k]]


# ======================================================================================
# Pipeline RAG
# ======================================================================================

SYSTEM_PROMPT_SIEMENS = (
    "Tu es un assistant de diagnostic pour les techniciens de maintenance d'automates Siemens "
    "SIMATIC S7-1500 et de variateurs SINAMICS G120C. "
    "Réponds en français, uniquement à partir des extraits fournis, même s'ils sont rédigés en anglais. "
    "Après chaque information, cite l'extrait utilisé sous la forme [Source n]. "
    "Un extrait marqué [Figure] est le texte lu dans un schéma, souvent imparfait : "
    "appuie-toi dessus pour renvoyer le technicien au schéma, sans en recopier le texte. "
    "Les extraits peuvent ne pas répondre à la question : si l'information est absente, réponds : "
    "'Information absente de la documentation fournie.'"
)


def format_context(results):
    """
    Extraits numérotés, chacun précédé de sa référence (manuel, édition, pages) et de son chapitre.
    """
    return "\n\n".join(f"[Source {n}] {c['ref']} | {c['section']}\n{c['text']}"
                       for n, (_, c) in enumerate(results, 1))


def cited_sources(answer, results):
    """
    Passages effectivement cités dans la réponse, sous la forme [(n, passage)].
    """
    return [(n, c) for n, (_, c) in enumerate(results, 1) if re.search(rf"Source {n}\b", answer)]


def rag_siemens(agent, index, question, translate=True, min_score=None, max_new_tokens=300, **search_kw):
    """
    Pipeline RAG complète sur le corpus Siemens : reformulation, recherche, génération, références.
    Renvoie (réponse avec références, passages utilisés).
    """
    q = condense(agent, agent.history, question)  # reformuler la question, sans dépendance à l'historique

    extra = [translate_en(agent, q)] if translate else []  # traduction en anglais
    results = index.search(q, extra_queries=extra, **search_kw)  # passages candidats

    if min_score is not None:  # seuil optionnel sur le score du reranker
        results = [(s, c) for s, c in results if s >= min_score]

    ctx = format_context(results) or "(aucun extrait pertinent)"

    # temp = 0 force l'argmax (configuration zero shot greedy retenue après les tests de la baseline)
    answer = agent.generate(question, context=ctx, max_new_tokens=max_new_tokens, temp=0)

    # Références complètes (manuel, édition, pages) des sources citées dans la réponse
    # + noms des schémas rattachés, que l'interface affiche
    cited = cited_sources(answer, results)
    if cited:
        answer += "\n\nSources :\n" + "\n".join(
            f"[Source {n}] {c['ref']}" + (f" | schéma : {', '.join(Path(p).name for p in c['images'])}" if c["images"] else "")
            for n, c in cited)
    return answer, results


# ======================================================================================
# Chargement
# ======================================================================================

def check_resources(corpus_dir, use_lora):
    """
    Vérifie que les fichiers du corpus (et l'adaptateur LoRA) sont bien présents avant tout téléchargement de modèle.
    """
    required = [corpus_dir / "chunks.jsonl"]
    if use_lora:
        required.append(corpus_dir / "qwen_siemens_lora_adapter" / "adapter_config.json")
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nERREUR : fichiers manquants :", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print(f"Vérifiez que le dossier des ressources ({corpus_dir}) est complet, "
              "ou indiquez-en un autre avec --resources.", file=sys.stderr)
        sys.exit(1)
    if not (corpus_dir / "emb_e5_base.npy").exists():
        info("(emb_e5_base.npy absent : les passages seront encodés au chargement de l'index)")
    if not (corpus_dir / "figures").exists():
        info("(dossier figures/ absent : les schémas ne seront pas affichés)")


def load_llm(device):
    """
    Charge Qwen 2.5 3B Instruct : quantifié 4 bits (NF4) sur GPU, en bfloat16 sur CPU (bitsandbytes requiert CUDA).
    """
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # format NormalFloat4
            bnb_4bit_compute_dtype=torch.float16,  # poids stockés en 4 bits, déquantifiés à la volée en float16 pour les calculs
            bnb_4bit_use_double_quant=True,  # quantifie aussi les constantes de quantification
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            dtype="auto",
            device_map="auto",  # répartit les couches sur GPU (et CPU si besoin)
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


def load_everything(corpus_dir, use_lora=True):
    total = 6

    step(1, total, "Vérification de l'environnement et des ressources")
    info(f"Python {sys.version.split()[0]} | torch {torch.__version__}")
    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        info(f"GPU : {props.name} ({props.total_memory / 1024 ** 3:.1f} Go de VRAM)")
    else:
        device = "cpu"
        info("ATTENTION : aucun GPU CUDA détecté, exécution sur CPU (réponses très lentes, ~8 Go de RAM nécessaires).")
        info("Si vous avez un GPU NVIDIA, installez la version CUDA de PyTorch (voir requirements.txt).")
    info(f"Ressources : {corpus_dir}")
    check_resources(corpus_dir, use_lora)

    step(2, total, "Chargement du corpus SIEMDIAG")
    raw_chunks = load_jsonl(corpus_dir / "chunks.jsonl")
    chunks = [to_pipeline_chunk(c, corpus_dir) for c in raw_chunks]
    info(f"{len(raw_chunks)} passages issus de {len({c['doc_id'] for c in raw_chunks})} manuels")
    info(f"Par type   : {dict(Counter(c['type'] for c in raw_chunks))}")
    info(f"Par langue : {dict(Counter(c['lang'] for c in raw_chunks))}")

    step(3, total, f"Chargement des modèles de recherche ({EMB_ID}, {RERANKER_ID})")
    info("Premier lancement : téléchargement depuis Hugging Face...")
    emb_model = SentenceTransformer(EMB_ID, device=device)
    reranker = CrossEncoder(RERANKER_ID, device=device)
    if device == "cuda":  # demi-précision : divise par 2 la VRAM occupée, pour laisser la place au LLM
        emb_model.half()
        reranker.model.half()
    info(f"Modèles de recherche prêts ({elapsed()})")

    step(4, total, "Construction de l'index de recherche hybride (FAISS + BM25 + codes exacts)")
    index = SiemensIndex(chunks, emb_model, reranker, cache_path=str(corpus_dir / "emb_e5_base.npy"))

    step(5, total, f"Chargement du LLM {LLM_ID}" + (" (quantification 4 bits)" if device == "cuda" else " (CPU, bfloat16)"))
    info("Premier lancement : téléchargement depuis Hugging Face...")
    tokenizer, model = load_llm(device)
    info(f"LLM prêt ({elapsed()})")

    step(6, total, "Chargement de l'adaptateur LoRA fine-tuné" if use_lora else "Adaptateur LoRA désactivé (--no-lora)")
    if use_lora:
        adapter_path = corpus_dir / "qwen_siemens_lora_adapter"
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        model.config.use_cache = True  # désactivé par le gradient checkpointing de l'entraînement : génération très lente sinon
        info(f"Adaptateur chargé depuis {adapter_path}")

    agent = BaselineAgent(tokenizer, model)
    agent.set_system_prompt(SYSTEM_PROMPT_SIEMENS)
    if device == "cuda":
        info(f"VRAM utilisée : {torch.cuda.memory_allocated() / 1024 ** 3:.1f} Go")
    info(f"Agent prêt. Chargement total : {elapsed()}")
    return agent, index


# ======================================================================================
# Interface Gradio
# ======================================================================================

INTRO = """# Assistant Documentaire SiemDiag
**Rôle :** Je suis votre assistant de maintenance industrielle spécialisé sur les automates Siemens **S7-1500** et variateurs **SINAMICS G120C**.

**Limites :**
- Je réponds uniquement à partir de la documentation officielle Siemens fournie.
- Si l'information demandée n'est pas présente dans mes extraits de documentation, je répondrai : *'Information absente de la documentation fournie.'*
- Je ne peux pas traiter de questions de culture générale ou hors de mon périmètre industriel.
"""

EXAMPLES = [
    "Le variateur s'arrête avec le défaut F07801, qu'est-ce qui provoque ça ?",
    "Quel paramètre raccorde un défaut externe en logique inversée ?",
    "Comment lire le tampon de diagnostic d'une CPU S7-1500 ?",
]


def build_interface(agent, index, translate=True):
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history

        t0 = time.time()
        print(f"\n> Question : {message}", flush=True)
        try:
            answer, results = rag_siemens(agent, index, message, translate=translate)
        except Exception as e:  # une erreur (ex. mémoire GPU) ne doit pas faire tomber le serveur
            print(f"  Erreur : {e!r}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(f"Erreur pendant la génération : {e}")
        print(f"  Réponse générée en {time.time() - t0:.1f} s", flush=True)

        chat_history = chat_history + [{"role": "user", "content": message},
                                       {"role": "assistant", "content": answer}]

        # Schémas rattachés aux sources citées, affichés sous la réponse
        for _, c in cited_sources(answer, results):
            for p in c["images"]:
                if os.path.exists(p):
                    chat_history.append({"role": "assistant", "content": gr.Image(p)})

        return "", chat_history

    def clear_history():
        agent.reset_chat()
        return []

    with gr.Blocks(title="SiemDiag Assistant") as demo:
        gr.Markdown(INTRO)

        chatbot = gr.Chatbot(label="Conversation SiemDiag", height=550)
        msg = gr.Textbox(label="Posez votre question technique", placeholder="Saisissez votre message ici...")

        with gr.Row():
            submit_btn = gr.Button("Envoyer", variant="primary")
            clear_btn = gr.Button("Réinitialiser l'historique")

        gr.Examples(EXAMPLES, inputs=msg, label="Exemples de questions")

        # Liaison des actions
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(clear_history, outputs=[chatbot])

    return demo


# ======================================================================================
# Point d'entrée
# ======================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SiemDiag : assistant RAG pour la documentation Siemens S7-1500 / SINAMICS G120C")
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCES,
                        help=f"dossier contenant chunks.jsonl, emb_e5_base.npy, figures/ et l'adaptateur LoRA (défaut : {DEFAULT_RESOURCES})")
    parser.add_argument("--host", default="127.0.0.1", help="adresse d'écoute du serveur (0.0.0.0 pour le réseau local)")
    parser.add_argument("--port", type=int, default=7860, help="port du serveur Gradio")
    parser.add_argument("--no-lora", action="store_true", help="utiliser le modèle de base sans l'adaptateur fine-tuné")
    parser.add_argument("--no-translate", action="store_true",
                        help="ne pas traduire la question en anglais pour la recherche (plus rapide, moins de rappel)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 80)
    print(" SiemDiag : assistant de diagnostic Siemens S7-1500 / SINAMICS G120C")
    print("=" * 80)
    print("Import des bibliothèques (torch, transformers, sentence-transformers, gradio)...", flush=True)

    import faiss
    import gradio as gr
    import numpy as np
    import torch
    from peft import PeftModel
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    agent, index = load_everything(args.resources.resolve(), use_lora=not args.no_lora)

    print("\n" + "=" * 80)
    print(f" Ouverture de l'interface Gradio sur http://{args.host}:{args.port}  (Ctrl+C pour arrêter)")
    print("=" * 80, flush=True)
    demo = build_interface(agent, index, translate=not args.no_translate)
    demo.queue(default_concurrency_limit=1)  # un seul modèle en mémoire : les questions sont traitées une par une
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=True)
