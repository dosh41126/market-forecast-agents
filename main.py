#!/usr/bin/env python3
import os
import sys
import threading
import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import bleach
import httpx
import pandas as pd
import aiosqlite
import weaviate
from weaviate.embedded import EmbeddedOptions
import yaml
import prometheus_client
import customtkinter as ctk
import tkinter as tk
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io import to_image
from openai import OpenAI

import numpy as np
from Pyfhel import Pyfhel, PyCtxt

# ─── Environment & Weaviate Setup ──────────────────────────────────────────────

try:
    MASTER_PASSWORD = os.environ["MASTER_PASSWORD"].encode()
    OPENAI_KEY      = os.environ["OPENAI_KEY"]
    AGENTS_FILE     = Path(os.environ.get("AGENTS_FILE", "agents.yaml"))
except KeyError as e:
    print(f"Missing required env var: {e.args[0]}", file=sys.stderr)
    sys.exit(1)

embedded_opts = EmbeddedOptions(
    persistence_data_path=os.path.expanduser("~/.local/share/weaviate"),
    binary_path=os.path.expanduser("~/.cache/weaviate-embedded"),
)
WEAVIATE_CLIENT = weaviate.Client(
    embedded_options=embedded_opts,
    additional_headers={"X-OpenAI-Api-Key": OPENAI_KEY},
)

# ─── FHE-Backed Vector Memory Module ────────────────────────────────────────────

class FHEManager:
    def __init__(self):
        self.HE = Pyfhel()                  
        self.HE.contextGen(p=2**15, m=2048)  # CKKS-like scheme
        self.HE.keyGen()

    def encrypt_vector(self, vector: List[float]) -> List[str]:
        """Encrypt each float into a base64‐encoded ciphertext."""
        enc = []
        for v in vector:
            ctxt = self.HE.encryptFrac(v)
            raw = ctxt.to_bytes()
            enc.append(base64.b64encode(raw).decode())
        return enc

    def decrypt_vector(self, ctxt_b64_list: List[str]) -> List[float]:
        """Decrypt a list of base64‐encoded ciphertexts back to floats."""
        dec = []
        for b64 in ctxt_b64_list:
            raw = base64.b64decode(b64)
            ctxt = PyCtxt(pyfhel=self.HE, serialized=raw)
            dec.append(self.HE.decryptFrac(ctxt))
        return dec

fhe_manager = FHEManager()

# ─── Agent Configuration Loading ───────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class AgentConfig:
    name: str
    description: str
    equation: str

with open(AGENTS_FILE, "r") as f:
    agents_yaml = yaml.safe_load(f)
AGENTS: List[AgentConfig] = [AgentConfig(**agent) for agent in agents_yaml.get("agents", [])]

# ─── Logging & Metrics ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
REQUEST_TIME = prometheus_client.Summary("debate_request_seconds", "Time spent in debate analysis")
prometheus_client.start_http_server(8000)

# ─── Key Management (AES-GCM with rotation & PBKDF2 backup) ────────────────────

BACKUP_PATH               = Path("aes_key_backup.bin")
USAGE_ROTATE_THRESHOLD    = 20
TIME_ROTATE_THRESHOLD_HOURS = 24
MAX_KEYS                  = 3

class KeyManager:
    def __init__(self, backup_path: Path = BACKUP_PATH):
        self.backup_path = backup_path
        self.salt: bytes
        self.keys: Dict[str, bytes] = {}
        self.current_id: Optional[str] = None
        self.usage_count: int = 0
        self.last_rotation: datetime
        self._lock = threading.Lock()
        self._load_or_init()

    def _derive_backup_key(self) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=200_000
        )
        return kdf.derive(MASTER_PASSWORD)

    def _load_or_init(self):
        if self.backup_path.exists():
            data = self.backup_path.read_bytes()
            self.salt = data[:16]
            nonce, ct = data[16:28], data[28:]
            try:
                plain = AESGCM(self._derive_backup_key()).decrypt(nonce, ct, None)
            except InvalidTag:
                raise RuntimeError("Backup file corrupt or tampered")
            obj = json.loads(plain.decode())
            for kid, b64 in obj["keys"].items():
                self.keys[kid] = base64.b64decode(b64)
            self.current_id   = obj["current_id"]
            self.usage_count  = obj.get("usage_count", 0)
            self.last_rotation= datetime.fromisoformat(obj["last_rotation"])
            logging.info(f"Loaded {len(self.keys)} keys; current_id={self.current_id}")
        else:
            self.salt = os.urandom(16)
            self._rotate_key(first=True)

    def _save_backup(self):
        obj = {
            "keys":         {kid: base64.b64encode(k).decode() for kid, k in self.keys.items()},
            "current_id":   self.current_id,
            "usage_count":  self.usage_count,
            "last_rotation":self.last_rotation.isoformat(),
        }
        plain     = json.dumps(obj).encode()
        kdf_key   = self._derive_backup_key()
        nonce     = os.urandom(12)
        ct        = AESGCM(kdf_key).encrypt(nonce, plain, None)
        self.backup_path.write_bytes(self.salt + nonce + ct)
        self.backup_path.chmod(0o600)

    def _prune_old_keys(self):
        if len(self.keys) <= MAX_KEYS:
            return
        for old in sorted(self.keys)[:len(self.keys) - MAX_KEYS]:
            del self.keys[old]

    def _rotate_key(self, first=False):
        new_key = os.urandom(32)
        if not first and self.current_id:
            prev    = self.keys[self.current_id]
            new_key = bytes(a ^ b for a, b in zip(new_key, prev))
        kid               = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.keys[kid]    = new_key
        self.current_id   = kid
        self.usage_count  = 0
        self.last_rotation= datetime.utcnow()
        self._prune_old_keys()
        self._save_backup()
        logging.info(f"Rotated key → {kid}")

    def maybe_rotate(self):
        with self._lock:
            now = datetime.utcnow()
            if (self.usage_count >= USAGE_ROTATE_THRESHOLD or
                (now - self.last_rotation) >= timedelta(hours=TIME_ROTATE_THRESHOLD_HOURS)):
                self._rotate_key()
            else:
                self._save_backup()

    def get_current_key(self) -> Tuple[str, bytes]:
        return self.current_id, self.keys[self.current_id]

    def get_key(self, kid: str) -> Optional[bytes]:
        return self.keys.get(kid)

key_manager = KeyManager()

def encrypt_txt(plain: str) -> str:
    kid, key = key_manager.get_current_key()
    aes       = AESGCM(key)
    nonce     = os.urandom(12)
    ct        = aes.encrypt(nonce, plain.encode(), None)
    payload   = {
        "key_id": kid,
        "nonce":  base64.b64encode(nonce).decode(),
        "ct":     base64.b64encode(ct).decode()
    }
    key_manager.usage_count += 1
    key_manager.maybe_rotate()
    return base64.b64encode(json.dumps(payload).encode()).decode()

def decrypt_txt(token: str) -> str:
    p   = json.loads(base64.b64decode(token).decode())
    key = key_manager.get_key(p["key_id"])
    if not key:
        raise ValueError(f"Key {p['key_id']} not found")
    aes   = AESGCM(key)
    nonce = base64.b64decode(p["nonce"])
    ct    = base64.b64decode(p["ct"])
    return aes.decrypt(nonce, ct, None).decode()

# ─── Storage & Encrypted Search Manager ────────────────────────────────────────

class StorageManager:
    DB_PATH   = Path("storage.db")
    W_CLIENT  = WEAVIATE_CLIENT
    EMB_MODEL = "text-embedding-ada-002"

    @classmethod
    async def init_db(cls):
        async with aiosqlite.connect(cls.DB_PATH) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ohlc_cache (
                    product_id TEXT NOT NULL,
                    granularity INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (product_id, granularity)
                );
            """)
            await db.commit()

        classes = [c["class"] for c in cls.W_CLIENT.schema.get()["classes"]]
        if "EncryptedContext" not in classes:
            cls.W_CLIENT.schema.create_class({
                "class": "EncryptedContext",
                "properties": [
                    {"name": "timestamp",   "dataType": ["date"]},
                    {"name": "content",     "dataType": ["text"]},
                    {"name": "enc_vector",  "dataType": ["text[]"]}
                ]
            })

    @classmethod
    async def save_context(cls, content: str):
        clean_content = bleach.clean(content)
        ts            = datetime.utcnow().isoformat()
        async with aiosqlite.connect(cls.DB_PATH) as db:
            await db.execute(
                "INSERT INTO context(timestamp,content) VALUES (?,?)",
                (ts, clean_content)
            )
            await db.commit()

        embedding = OpenAI(api_key=OPENAI_KEY) \
            .embeddings.create(model=cls.EMB_MODEL, input=[clean_content]) \
            .data[0].embedding

        # Encrypt the embedding under FHE
        enc_embedding = fhe_manager.encrypt_vector(embedding)

        cls.W_CLIENT.data_object.create(
            {"timestamp": ts, "content": clean_content, "enc_vector": enc_embedding},
            "EncryptedContext"
        )

    @classmethod
    async def recent_contexts(cls, limit: int = 5) -> List[str]:
        async with aiosqlite.connect(cls.DB_PATH) as db:
            cur  = await db.execute(
                "SELECT content FROM context ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = await cur.fetchall()
            return [r[0] for r in rows]

    @classmethod
    async def get_cached_ohlc(cls, product_id: str, gran_sec: int) -> Optional[pd.DataFrame]:
        async with aiosqlite.connect(cls.DB_PATH) as db:
            cur = await db.execute(
                "SELECT fetched_at, data_json FROM ohlc_cache WHERE product_id=? AND granularity=?",
                (product_id, gran_sec)
            )
            row = await cur.fetchone()
            if not row:
                return None
            fetched_at, data_json = row
            if datetime.utcnow() - datetime.fromisoformat(fetched_at) < timedelta(seconds=gran_sec):
                return pd.read_json(data_json, convert_dates=["time"]).sort_values("time")
            return None

    @classmethod
    async def cache_ohlc(cls, product_id: str, gran_sec: int, df: pd.DataFrame):
        js = df.to_json(date_format="iso")
        ts = datetime.utcnow().isoformat()
        async with aiosqlite.connect(cls.DB_PATH) as db:
            await db.execute("""
                INSERT INTO ohlc_cache(product_id, granularity, fetched_at, data_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(product_id,granularity) DO UPDATE
                  SET fetched_at=excluded.fetched_at,
                      data_json=excluded.data_json
            """, (product_id, gran_sec, ts, js))
            await db.commit()

    @classmethod
    async def search_contexts(cls, query: str, limit: int = 5) -> List[str]:
        """Embed `query`, fetch encrypted vectors in bulk, decrypt, then rank by cosine similarity."""
        embedding = OpenAI(api_key=OPENAI_KEY) \
            .embeddings.create(model=cls.EMB_MODEL, input=[query]) \
            .data[0].embedding

        # Fetch a batch of encrypted contexts
        resp = cls.W_CLIENT.query \
            .get("EncryptedContext", ["content", "enc_vector"]) \
            .with_limit(100) \
            .do()

        candidates: List[Tuple[float, str]] = []
        for obj in resp["data"]["Get"]["EncryptedContext"]:
            enc_vec = obj["enc_vector"]
            # Decrypt FHE‐ciphertext back to plaintext vector
            decrypted = fhe_manager.decrypt_vector(enc_vec)
            # Compute cosine similarity
            sim = np.dot(embedding, decrypted) / (np.linalg.norm(embedding) * np.linalg.norm(decrypted))
            candidates.append((sim, obj["content"]))

        # Sort and return top‐N
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:limit]]

# ─── OHLC Fetch + Cache Logic ─────────────────────────────────────────────────

async def fetch_ohlc(product_id="ETH-USD", gran_sec=4*3600) -> pd.DataFrame:
    cached = await StorageManager.get_cached_ohlc(product_id, gran_sec)
    if cached is not None:
        return cached

    now   = datetime.utcnow()
    start = now - timedelta(seconds=gran_sec * 50)
    params = {
        "start":     start.isoformat(),
        "end":       now.isoformat(),
        "granularity": gran_sec
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params=params
        )
        resp.raise_for_status()
        data = resp.json()

    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")

    await StorageManager.cache_ohlc(product_id, gran_sec, df)
    return df.sort_values("time")

# ─── Prompt Builders & Sharded Debate Orchestration ─────────────────────────────

def build_prompts_for_agents(
    agents: List[AgentConfig],
    chart_b64: str,
    market_context: str
) -> List[Dict]:
    """
    Constructs the full hypertime debate prompts—including System, per‐agent analysis,
    Debate, Revision, and Consensus stages—with equations and 3,000+ word requirements.
    """
    # 1) System Instruction (identical for every shard)
    prompts = [{
        "role": "system",
        "content": (
            "You are an advanced hypertime orchestrator coordinating multiple GPT-4o agents, "
            "each running a unique forecasting model. Follow this pipeline strictly:\n\n"
            "  1. Individual Analyses: Each agent produces ≥3,000 words covering their model:\n"
            "     • Core theory (quantum mechanics, nonlinear dynamics, evolutionary loops, sentiment hypercycles, planetary consensus).\n"
            "     • Parameter sensitivity analysis (list each α, β, γ, δ, etc. and its impact).\n"
            "     • Backtest versus live deviation quantification (include statistical tables if needed).\n"
            "     • Enumerate ≥3 distinct failure modes with root‐cause paths.\n\n"
            "  2. Debate Stage: Agents critique peer outputs, citing exact equations, parameter values, "
            "and risk scenarios (market shocks, black swan, protocol forks).\n\n"
            "  3. Revision Stage: Each agent revises its forecast with updated confidence intervals, "
            "parameter recalibration, and emergent hypertime error corrections.\n\n"
            "  4. Consensus Synthesis: Merge all revised outputs into one directive (LONG/SHORT/NEUTRAL) "
            "with ≥5 technical bullet points, a confidence percentage, and a hypertime risk matrix.\n\n"
            "Enclose ONLY the final consensus synthesis within [replytemplate]...[/replytemplate] tags."
        )
    }]

    # 2) Per-Agent Analysis Prompts
    for agent in agents:
        if agent.name.upper() == "A":
            content = (
                f"Agent A (Quantum Entanglement Market Equation — QEME):\n\n"
                f"Equation:\n"
                f"    P_crypto(t) = α · QFT_macro(t) + β · EM_chain(t) + γ · S_sentiment(t) "
                f"+ δ · √V_vol(t) + λ · Ψ_entangle(t)\n\n"
                "Task:\n"
                "    Produce at least 3,000 words analyzing:\n"
                "      • Quantum Fourier transforms on macro drivers (derive QFT_macro(t)).\n"
                "      • State‐vector entanglement transitions Ψ_entangle and decoherence error correction.\n"
                "      • Hyperbolic parameter mapping for α, β, γ, δ, λ under high‐volatility regimes.\n"
                "      • Backtest vs. live deviations across 10 datasets (include variance tables).\n"
                "      • Enumerate ≥3 failure modes (e.g., qubit decoherence spikes, sentiment collision, chain reorgs).\n"
                f"Market context: {market_context}\n"
            )
        elif agent.name.upper() == "B":
            content = (
                f"Agent B (Multiversal Feedback Attractor — MFA):\n\n"
                f"Equation:\n"
                f"    P(t+1) = θ₁ · P(t) + θ₂ · P_alt(t) + ε · ∫₀ᵀ f_chaos(τ) dτ + ζ · GAI(t)\n\n"
                "Task:\n"
                "    Produce at least 3,000 words simulating:\n"
                "      • Six parallel universe forks P_alt(t) and their feedback loops.\n"
                "      • Nonlinear chaos integrals ∫₀ᵀ f_chaos and sensitivity to ε.\n"
                "      • Synthetic GAI(t) shock injections and timeline entanglement.\n"
                "      • Peer‐review critique of Agent A’s entanglement assumptions.\n"
                "      • ≥3 failure modes (chaotic divergence, attractor bifurcation, meta‐AI misalignment).\n"
                f"Market context: {market_context}\n"
            )
        elif agent.name.upper() == "C":
            content = (
                f"Agent C (Recursive Evolutionary Market Driver — REMD):\n\n"
                f"Equation:\n"
                f"    P(t+1) = P(t) + ∑₍ᵢ₌₁₎ⁿ ωᵢ · Fᵢ(t) + κ · AI_meta(t)\n\n"
                "Task:\n"
                "    Produce at least 3,000 words detailing:\n"
                "      • Recursive learning loops across Fᵢ(t) features.\n"
                "      • Evolutionary path dependency and hysteresis effects.\n"
                "      • Hypertime error correction cycles κ · AI_meta(t).\n"
                "      • Cross‐model drift critique (Agents A & B).\n"
                "      • ≥3 failure modes (recursive runaway, feature overfitting, meta‐model conflict).\n"
                f"Market context: {market_context}\n"
            )
        elif agent.name.upper() == "D":
            content = (
                f"Agent D (Techno-Social Supercycle Model — TSSM):\n\n"
                f"Equation:\n"
                f"    P_crypto(t) = σ · log(N_users(t)) + τ · HashRate(t) + φ · ΔReg(t) + η · Macro_global(t)\n\n"
                "Task:\n"
                "    Produce at least 3,000 words covering:\n"
                "      • Logarithmic social adoption cycles.\n"
                "      • Network throughput and hashrate dynamics (τ tuning).\n"
                "      • Regulatory delta shifts ΔReg(t) and policy shock scenarios.\n"
                "      • Global macro influence integrals η · Macro_global(t).\n"
                "      • Critiques of Agents A, B, C hypertime assumptions.\n"
                "      • ≥3 failure modes (regulatory black swan, adoption plateau, network congestion).\n"
                f"Market context: {market_context}\n"
            )
        elif agent.name.upper() == "E":
            content = (
                f"Agent E (Hypercycle Sentiment-VIX Fusion):\n\n"
                f"Equation:\n"
                f"    P_future(t) = μ · Sentiment_long(t) + ν · VIX_crypto(t) + ξ · ∫ₜ₀ᵗ Innovation_rate(τ) dτ\n\n"
                "Task:\n"
                "    Produce at least 3,000 words analyzing:\n"
                "      • Long‐term sentiment trend modeling μ · Sentiment_long(t).\n"
                "      • Volatility index fusion ν · VIX_crypto(t).\n"
                "      • Innovation‐pulse integration ∫ₜ₀ᵗ Innovation_rate(τ).\n"
                "      • Debate of prior parameter choices.\n"
                "      • ≥3 failure modes (sentiment inversion, VIX decoupling, innovation shock underestimation).\n"
                f"Market context: {market_context}\n"
            )
        elif agent.name.upper() == "F":
            content = (
                f"Agent F (Nosonar Global Market Equation):\n\n"
                f"Equation:\n"
                f"    P_humanity(t) = ∑₍ₖ₌₁₎ⁿ ωₖ · Dₖ(t) + θ · MetaSentiment_planet(t) "
                f"+ γ · Innovation_pulse(t) + ξ · GlobalConsensus(t)\n\n"
                "Task:\n"
                "    Produce at least 3,000 words synthesizing:\n"
                "      • Planetary-scale demand drivers Dₖ(t).\n"
                "      • MetaSentiment_planet metrics and γ innovation pulses.\n"
                "      • Global consensus formation ξ · GlobalConsensus(t).\n"
                "      • Critique all prior agents’ edge cases.\n"
                "      • ≥3 failure modes (geo-political fracturing, data collapse, consensus deadlock).\n"
                f"Market context: {market_context}\n"
            )
        else:
            # Fallback generic prompt
            content = (
                f"Agent {agent.name} ({agent.description}):\n"
                f"Equation: {agent.equation}\n"
                "Task: Detailed 3,000+ word analysis including methodology, sensitivity, backtest vs. live, "
                "and ≥3 failure modes.\n"
                f"Market context: {market_context}\n"
            )

        prompts.append({
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        })

    # 3) Debate, Revision, Consensus Stages
    prompts.extend([
        {"role": "user", "content": "DEBATE STAGE: Each agent must cite peers’ equations, parameter values, and enumerate ≥3 failure modes."},
        {"role": "user", "content": "REVISION STAGE: Agents update forecasts with revised confidence intervals, parameter adjustments, and emergent hypertime corrections."},
        {"role": "user", "content": "CONSENSUS SYNTHESIS: Merge revised outputs into one directive (LONG/SHORT/NEUTRAL) with ≥5 technical bullets, confidence %, and a hypertime risk matrix. Enclose within [replytemplate] tags."}
    ])

    return prompts

async def debate_shard(chart_b64: str, market_ctx: str, shard_agents: List[AgentConfig]) -> str:
    prompts = build_prompts_for_agents(shard_agents, chart_b64, market_ctx)
    client  = OpenAI(api_key=OPENAI_KEY)
    resp    = await client.chat.completions.create_async(
        model="gpt-4o-mini",
        messages=prompts,
        max_tokens=32000,
        stream=True
    )
    collected = ""
    async for chunk in resp:
        delta = chunk.choices[0].delta.get("content")
        if delta:
            collected += delta
    return collected

@REQUEST_TIME.time()
async def debate_and_consensus(chart_b64: str, market_ctx: str) -> str:
    # Determine number of shards (up to CPU count)
    cpu = os.cpu_count() or 4
    num_shards = min(cpu, len(AGENTS))
    # Round‐robin partition into shards
    shards = [AGENTS[i::num_shards] for i in range(num_shards)]

    # Run each shard in parallel
    partials = await asyncio.gather(*[
        debate_shard(chart_b64, market_ctx, shard) for shard in shards
    ])

    # Aggregator stage: merge partial syntheses into final consensus
    agg_prompts = [
        {
            "role": "system",
            "content": (
                "You are a hypertime aggregator. You have received multiple partial consensus texts "
                "from parallel shards. Merge them into a single final consensus synthesis, following "
                "the original pipeline: directive (LONG/SHORT/NEUTRAL), ≥5 technical bullets, confidence %, "
                "and a hypertime risk matrix."
            )
        },
        {
            "role": "user",
            "content": "\n\n---\n\n".join(partials)
        }
    ]
    client = OpenAI(api_key=OPENAI_KEY)
    agg_resp = await client.chat.completions.create_async(
        model="gpt-4o-mini",
        messages=agg_prompts,
        max_tokens=32000,
        stream=True
    )
    final = ""
    async for chunk in agg_resp:
        delta = chunk.choices[0].delta.get("content")
        if delta:
            final += delta
    return final

# ─── CustomTkinter Dashboard ─────────────────────────────────────────────────

class AdvancedDashboard(ctk.CTk):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self.title("Hypertime Mesh Consensus")
        self.geometry("1280x1100")
        self.configure(padx=20, pady=20)

        asyncio.run_coroutine_threadsafe(StorageManager.init_db(), loop)

        ctrl = ctk.CTkFrame(self); ctrl.pack(fill="x", pady=(0,10))
        self.sym  = ctk.StringVar(value="ETH-USD")
        self.gran = ctk.StringVar(value="4h")
        ctk.CTkOptionMenu(ctrl, variable=self.sym,  values=["ETH-USD","BTC-USD","SOL-USD"]).grid(row=0,col=0,padx=5)
        ctk.CTkOptionMenu(ctrl, variable=self.gran, values=["4h","5h"]).grid(row=0,col=1,padx=5)
        ctk.CTkButton(ctrl, text="Refresh", command=lambda: self.schedule(True)).grid(row=0,col=2,padx=5)

        self.chart_lbl = ctk.CTkLabel(self, text=""); self.chart_lbl.pack(pady=10)
        self.txt       = ctk.CTkTextbox(self, width=1200, height=460, font=("Consolas",13), wrap="word")
        self.txt.configure(border_color="#444444", fg_color="#222222"); self.txt.pack(pady=10)
        self.status    = ctk.CTkLabel(self, text="Ready", text_color="gray"); self.status.pack(pady=5)

        self.after(1000, lambda: self.schedule(False))
        self.after(180000, lambda: self.schedule(False))

    def schedule(self, force: bool):
        self.status.configure(text="Scheduling update…")
        fut = asyncio.run_coroutine_threadsafe(self._update(force), self.loop)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut: asyncio.Future):
        if err := fut.exception():
            logging.exception("Update failed", exc_info=err)
            self.txt.delete("0.0","end")
            self.txt.insert("0.0", f"Error: {err}")

    async def _update(self, force: bool):
        self.status.configure(text="Fetching OHLC…")
        pid, gran = self.sym.get(), self.gran.get()
        gran_s    = 4*3600 if gran=="4h" else 5*3600

        df = await fetch_ohlc(pid, gran_s)
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color="yellow", decreasing_line_color="white",
            increasing_line_width=1.5, decreasing_line_width=1.5
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema20"], name="EMA20", line=dict(width=2)), row=1, col=1)
        fig.update_layout(template="plotly_dark", height=600,
                          margin=dict(l=40,r=40,t=60,b=40),
                          title=dict(text=f"{pid} {gran.upper()} Candles", x=0.5),
                          xaxis=dict(gridcolor="gray", gridwidth=0.2, title="Time"),
                          yaxis=dict(gridcolor="gray", gridwidth=0.2, title="Price (USD)"))

        img = to_image(fig, format="png")
        b64 = base64.b64encode(img).decode()
        photo = tk.PhotoImage(data=b64)
        chart_img = ctk.CTkImage(light_image=photo, size=(1200,600))
        self.chart_lbl.configure(image=chart_img); self.chart_lbl.image = chart_img

        last = df["close"].iloc[-1]
        ctx  = f"{pid}=${last:.2f}, Range {df['low'].min():.2f}-{df['high'].max():.2f} {gran.upper()}"
        await StorageManager.save_context(ctx)

        recent  = await StorageManager.recent_contexts()
        similar = await StorageManager.search_contexts(bleach.clean(ctx), limit=5)
        past_ctx= recent + similar

        self.status.configure(text="Running hypertime mesh debate…")
        result = await debate_and_consensus(b64, ctx)

        if "[replytemplate]" in result and "[/replytemplate]" in result:
            start = result.index("[replytemplate]") + len("[replytemplate]")
            end   = result.index("[/replytemplate]")
            out   = result[start:end].strip()
        else:
            out = (result[:9000] + "...\n[Truncated]") if len(result) > 9000 else result

        self.txt.delete("0.0","end"); self.txt.insert("0.0", out)
        self.status.configure(text="Complete.")

def main():
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    AdvancedDashboard(loop).mainloop()

if __name__ == "__main__":
    main()
