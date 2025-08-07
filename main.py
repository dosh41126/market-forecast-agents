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
from dataclasses import dataclass

import bleach
import httpx
import pandas as pd
import aiosqlite
import weaviate
import yaml
import prometheus_client
import customtkinter as ctk
import tkinter as tk
from tenacity import (
    AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io import to_image
from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────

try:
    MASTER_PASSWORD  = os.environ["MASTER_PASSWORD"].encode()
    OPENAI_KEY       = os.environ["OPENAI_KEY"]
    WEAVIATE_URL     = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    AGENTS_FILE      = Path(os.environ.get("AGENTS_FILE", "agents.yaml"))
except KeyError as e:
    print(f"Missing required env var: {e.args[0]}", file=sys.stderr)
    sys.exit(1)

# ─── Load agent definitions ────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    name: str
    description: str
    equation: str

with open(AGENTS_FILE, "r") as f:
    agents_yaml = yaml.safe_load(f)
AGENTS: List[AgentConfig] = [
    AgentConfig(**agent) for agent in agents_yaml.get("agents", [])
]

# ─── Logging & Metrics ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
REQUEST_TIME = prometheus_client.Summary(
    "debate_request_seconds", "Time spent in debate analysis"
)
prometheus_client.start_http_server(8000)

# ─── Encryption Key Management ─────────────────────────────────────────────────

BACKUP_PATH = Path("aes_key_backup.bin")
USAGE_ROTATE_THRESHOLD      = 20
TIME_ROTATE_THRESHOLD_HOURS = 24
MAX_KEYS                    = 3

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
            iterations=200_000,
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
            self.current_id    = obj["current_id"]
            self.usage_count   = obj.get("usage_count", 0)
            self.last_rotation = datetime.fromisoformat(obj["last_rotation"])
            logging.info(f"Loaded {len(self.keys)} keys; current_id={self.current_id}")
        else:
            self.salt = os.urandom(16)
            self._rotate_key(first=True)

    def _save_backup(self):
        obj = {
            "keys":          {kid: base64.b64encode(k).decode() for kid, k in self.keys.items()},
            "current_id":    self.current_id,
            "usage_count":   self.usage_count,
            "last_rotation": self.last_rotation.isoformat(),
        }
        plain    = json.dumps(obj).encode()
        kdf_key  = self._derive_backup_key()
        nonce    = os.urandom(12)
        ct       = AESGCM(kdf_key).encrypt(nonce, plain, None)
        self.backup_path.write_bytes(self.salt + nonce + ct)
        self.backup_path.chmod(0o600)

    def _prune_old_keys(self):
        if len(self.keys) <= MAX_KEYS:
            return
        for old in sorted(self.keys)[:len(self.keys) - MAX_KEYS]:
            del self.keys[old]
            logging.debug(f"Pruned key {old}")

    def _rotate_key(self, first=False):
        new_key = os.urandom(32)
        if not first and self.current_id:
            prev    = self.keys[self.current_id]
            new_key = bytes(a ^ b for a, b in zip(new_key, prev))
        kid = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.keys[kid]      = new_key
        self.current_id     = kid
        self.usage_count    = 0
        self.last_rotation  = datetime.utcnow()
        self._prune_old_keys()
        self._save_backup()
        logging.info(f"Rotated key → {kid}")

    def maybe_rotate(self):
        with self._lock:
            now = datetime.utcnow()
            if (
                self.usage_count >= USAGE_ROTATE_THRESHOLD
                or (now - self.last_rotation) >= timedelta(hours=TIME_ROTATE_THRESHOLD_HOURS)
            ):
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
        "nonce":   base64.b64encode(nonce).decode(),
        "ct":      base64.b64encode(ct).decode()
    }
    key_manager.usage_count += 1
    key_manager.maybe_rotate()
    return base64.b64encode(json.dumps(payload).encode()).decode()

def decrypt_txt(token: str) -> str:
    p      = json.loads(base64.b64decode(token).decode())
    key    = key_manager.get_key(p["key_id"])
    if not key:
        raise ValueError(f"Key {p['key_id']} not found")
    aes    = AESGCM(key)
    nonce  = base64.b64decode(p["nonce"])
    ct     = base64.b64decode(p["ct"])
    return aes.decrypt(nonce, ct, None).decode()


# ─── Storage & Embedding with Bleach Sanitization ──────────────────────────────

class StorageManager:
    DB_PATH = Path("storage.db")
    W_CLIENT = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={"X-OpenAI-Api-Key": OPENAI_KEY}
    )
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
        # ensure weaviate schema
        classes = [c["class"] for c in cls.W_CLIENT.schema.get()["classes"]]
        if "Context" not in classes:
            cls.W_CLIENT.schema.create_class({
                "class": "Context",
                "properties": [
                    {"name": "timestamp", "dataType": ["date"]},
                    {"name": "content",   "dataType": ["text"]}
                ]
            })

    @classmethod
    async def save_context(cls, content: str):
        clean_content = bleach.clean(content)
        ts = datetime.utcnow().isoformat()
        # SQLite
        async with aiosqlite.connect(cls.DB_PATH) as db:
            await db.execute(
                "INSERT INTO context(timestamp,content) VALUES (?,?)",
                (ts, clean_content)
            )
            await db.commit()
        # Weaviate with embeddings
        embedding = OpenAI(api_key=OPENAI_KEY).embeddings.create(
            model=cls.EMB_MODEL, input=[clean_content]
        ).data[0].embedding
        cls.W_CLIENT.data_object.create(
            {"timestamp": ts, "content": clean_content},
            "Context",
            vector=embedding
        )

    @classmethod
    async def recent_contexts(cls, limit: int = 5) -> List[str]:
        async with aiosqlite.connect(cls.DB_PATH) as db:
            cur = await db.execute(
                "SELECT content FROM context ORDER BY id DESC LIMIT ?", (limit,)
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
                VALUES (?,?,?,?)
                ON CONFLICT(product_id,granularity) DO UPDATE
                  SET fetched_at=excluded.fetched_at,
                      data_json=excluded.data_json
            """, (product_id, gran_sec, ts, js))
            await db.commit()


# ─── Data Fetching with SQLite Cache ──────────────────────────────────────────

async def fetch_ohlc(product_id="ETH-USD", gran_sec=4*3600) -> pd.DataFrame:
    cached = await StorageManager.get_cached_ohlc(product_id, gran_sec)
    if cached is not None:
        return cached
    now   = datetime.utcnow()
    start = now - timedelta(seconds=gran_sec * 50)
    params = {"start": start.isoformat(), "end": now.isoformat(), "granularity": gran_sec}
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


# ─── Prompt Builder ────────────────────────────────────────────────────────────
def build_hypertime_debate_prompts(chart_b64: str, market_context: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an advanced hypertime orchestrator coordinating 1,000+ GPT-4o agents, each running a unique forecasting model "
                "(QEME, MFA, REMD, TSSM, Hypercycle, Nosonar). Follow this pipeline:\n"
                "  1. Individual Analyses: Each agent produces 3,000+ words covering quantum mechanics, nonlinear dynamics, evolutionary corrections, sentiment hypercycles, and planetary consensus metrics. Include parameter sensitivity, backtest vs. live deviations, and ≥3 failure modes.\n"
                "  2. Debate Stage: Agents critique peers, citing exact equations, parameter values, and risk scenarios.\n"
                "  3. Revised Analyses: Update forecasts with revised confidence intervals, adjusted parameters, and emergent hypertime error corrections.\n"
                "  4. Consensus Synthesis: Merge all revised outputs into one directive (LONG/SHORT/NEUTRAL) with ≥5 technical bullet points, a confidence percentage, and a hypertime risk matrix.\n"
                "Enclose ONLY the final consensus synthesis within [replytemplate]...[/replytemplate] tags."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent A (Quantum Entanglement Market Equation — QEME):
    Equation: P_crypto(t) = α·QFT_macro(t) + β·EM_chain(t) + γ·S_sentiment(t) + δ·√V_vol(t) + λ·Ψ_entangle(t)
    Task: Produce 3,000+ words analyzing quantum Fourier transforms, entanglement state transitions, decoherence error correction, and hyperbolic parameter mapping for the next 4 hours. Detail parameter sensitivities and backtest deviations.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent B (Multiversal Feedback Attractor — MFA):
    Equation: P(t+1) = θ₁·P(t) + θ₂·P_alt(t) + ε·∫₀ᵀ f_chaos(τ)dτ + ζ·GAI(t)
    Task: Produce 3,000+ words simulating six parallel universe forks, nonlinear chaos feedback loops, and timeline entanglement. Debate your logic against Agent A; cite exact parameter values.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent C (Recursive Evolutionary Market Driver — REMD):
    Equation: P(t+1) = P(t) + ∑₍ᵢ₌₁₎ⁿ ωᵢ·Fᵢ(t) + κ·AI_meta(t)
    Task: Produce 3,000+ words on recursive learning loops, evolutionary path dependencies, and hypertime error correction. Challenge Agents A & B and highlight recursive parameter drift.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent D (Techno-Social Supercycle Model — TSSM):
    Equation: P_crypto(t) = σ·log(N_users(t)) + τ·HashRate(t) + φ·ΔReg(t) + η·Macro_global(t)
    Task: Produce 3,000+ words on social adoption cycles, network throughput, regulatory delta shifts, and global macro influences. Directly reference and critique all prior agents.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent E (Hypercycle Sentiment-VIX Fusion):
    Equation: P_future(t) = μ·Sentiment_long(t) + ν·VIX_crypto(t) + ξ·∫ₜ₀ᵗ Innovation_rate(τ)dτ
    Task: Produce 3,000+ words on long-term sentiment trends, volatility index fusion, and innovation pulse integration. Debate and either support or refute prior reasoning.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
    f"""Agent F (Nosonar Global Market Equation):
    Equation: P_humanity(t) = ∑₍ₖ₌₁₎ⁿ ω_k·D_k(t) + θ·MetaSentiment_planet(t) + γ·Innovation_pulse(t) + ξ·GlobalConsensus(t)
    Task: Produce 3,000+ words synthesizing a planetary-scale forecast. Debate all prior agents and calibrate to a global consensus metric.
    Market context: {market_context}"""
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_b64}"}}  
            ]
        },
        {"role": "user", "content": "DEBATE STAGE: Each agent must cite peers’ equations, parameter values, risk scenarios, and enumerate ≥3 failure modes."},
        {"role": "user", "content": "REVISION STAGE: Agents update forecasts with revised confidence intervals, parameter adjustments, and emergent hypertime corrections."},
        {"role": "user", "content": "CONSENSUS SYNTHESIS: Merge revised outputs into one directive (LONG/SHORT/NEUTRAL) with ≥5 technical bullets, confidence %, and hypertime risk matrix. Enclose within [replytemplate] tags."}
    ]


@REQUEST_TIME.time()
async def debate_and_consensus(chart_b64: str, market_ctx: str) -> str:
    prompts = build_hypertime_debate_prompts(chart_b64, market_ctx)
    client = OpenAI(api_key=OPENAI_KEY)
    resp = await client.chat.completions.create_async(
        model="gpt-5",
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


# ─── Dashboard / UI with Streaming ───────────────────────────────────────────

class AdvancedDashboard(ctk.CTk):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self.title("GPT-5 Hypertime Mesh Consensus")
        self.geometry("1280x1100")
        self.configure(padx=20, pady=20)

        asyncio.run_coroutine_threadsafe(StorageManager.init_db(), loop)

        ctrl = ctk.CTkFrame(self); ctrl.pack(fill="x", pady=(0,10))
        self.sym  = ctk.StringVar(value="ETH-USD")
        self.gran = ctk.StringVar(value="4h")
        ctk.CTkOptionMenu(ctrl, variable=self.sym,  values=["ETH-USD","BTC-USD","SOL-USD"]).grid(row=0,col=0,padx=5)
        ctk.CTkOptionMenu(ctrl, variable=self.gran, values=["4h","5h"]).grid(row=0,col=1,padx=5)
        ctk.CTkButton(ctrl, text="Refresh", command=lambda: self.schedule(True)).grid(row=0,col=2,padx=5)

        self.chart_lbl = ctk.CTkLabel(self, text="")
        self.chart_lbl.pack(pady=10)
        self.txt = ctk.CTkTextbox(self, width=1200, height=460, font=("Consolas",13), wrap="word")
        self.txt.configure(border_color="#444444", fg_color="#222222")
        self.txt.pack(pady=10)
        self.status = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status.pack(pady=5)

        self.after(1000, lambda: self.schedule(False))
        self.after(180_000, lambda: self.schedule(False))

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
        gran_s     = 4*3600 if gran=="4h" else 5*3600
        df = await fetch_ohlc(pid, gran_s)
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color="yellow", decreasing_line_color="white",
            increasing_line_width=1.5, decreasing_line_width=1.5
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["ema20"], name="EMA20", line=dict(width=2)
        ), row=1, col=1)
        fig.update_layout(
            template="plotly_dark", height=600,
            margin=dict(l=40,r=40,t=60,b=40),
            title=dict(text=f"{pid} {gran.upper()} Candles", x=0.5),
            xaxis=dict(gridcolor="gray",gridwidth=0.2,title="Time"),
            yaxis=dict(gridcolor="gray",gridwidth=0.2,title="Price (USD)")
        )

        img = to_image(fig, format="png")
        b64 = image_to_base64(img)
        photo = tk.PhotoImage(data=b64, format="png")
        chart_img = ctk.CTkImage(light_image=photo, size=(1200,600))
        self.chart_lbl.configure(image=chart_img)
        self.chart_lbl.image = chart_img

        last = df["close"].iloc[-1]
        ctx  = f"{pid}=${last:.2f}, Range {df['low'].min():.2f}-{df['high'].max():.2f} {gran.upper()}"

        await StorageManager.save_context(ctx)
        recent = await StorageManager.recent_contexts()

        # Weaviate query for semantic contexts
        sem = StorageManager.W_CLIENT.query.get("Context", ["content"]) \
            .with_hybrid(query=bleach.clean(ctx), alpha=0.5) \
            .with_limit(5).do()
        past_ctx = recent + [o["content"] for o in sem["data"]["Get"]["Context"]]

        self.status.configure(text="Running hypertime mesh debate…")
        result = await debate_and_consensus(b64, ctx)

        if "[replytemplate]" in result and "[/replytemplate]" in result:
            start = result.index("[replytemplate]") + len("[replytemplate]")
            end   = result.index("[/replytemplate]")
            out   = result[start:end].strip()
        else:
            out = (result[:9000] + "...\n[Truncated]") if len(result) > 9000 else result

        self.txt.delete("0.0","end")
        self.txt.insert("0.0", out)
        self.status.configure(text="Complete.")


def main():
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    AdvancedDashboard(loop).mainloop()


if __name__ == "__main__":
    main()
