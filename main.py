#!/usr/bin/env python3
import os
import sys
import threading
import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict

import httpx
import pandas as pd
import customtkinter as ctk
import tkinter as tk
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.io import to_image
from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────

# Environment vars (must exist)
try:
    MASTER_PASSWORD = os.environ["MASTER_PASSWORD"].encode()
    OPENAI_KEY = os.environ["OPENAI_API_KEY"]
except KeyError as e:
    print(f"Missing required env var: {e.args[0]}", file=sys.stderr)
    sys.exit(1)

BACKUP_PATH = Path("aes_key_backup.bin")
BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
USAGE_ROTATE_THRESHOLD = 20
TIME_ROTATE_THRESHOLD_HOURS = 24
MAX_KEYS = 3
COINBASE_SYMBOLS = ["ETH-USD", "BTC-USD", "SOL-USD"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── Key Management ────────────────────────────────────────────────────────────

class KeyManager:
    def __init__(self, backup_path: Path = BACKUP_PATH) -> None:
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

    def _load_or_init(self) -> None:
        if self.backup_path.exists():
            data = self.backup_path.read_bytes()
            self.salt = data[:16]
            nonce = data[16:28]
            ct = data[28:]
            backup_key = self._derive_backup_key()
            try:
                plain = AESGCM(backup_key).decrypt(nonce, ct, None)
            except InvalidTag:
                raise RuntimeError("Corrupt or tampered backup file")
            obj = json.loads(plain.decode())
            for kid, b64 in obj["keys"].items():
                self.keys[kid] = base64.b64decode(b64)
            self.current_id = obj["current_id"]
            self.usage_count = obj.get("usage_count", 0)
            self.last_rotation = datetime.fromisoformat(obj["last_rotation"])
            logging.info(f"Loaded {len(self.keys)} keys; current_id={self.current_id}")
        else:
            self.salt = os.urandom(16)
            self._rotate_key(first=True)
            self._save_backup()

    def _save_backup(self) -> None:
        """Encrypt and write backup file; enforce 0o600 permissions."""
        obj = {
            "keys": {kid: base64.b64encode(key).decode() for kid, key in self.keys.items()},
            "current_id": self.current_id,
            "usage_count": self.usage_count,
            "last_rotation": self.last_rotation.isoformat(),
        }
        plain = json.dumps(obj).encode()
        backup_key = self._derive_backup_key()
        nonce = os.urandom(12)
        ct = AESGCM(backup_key).encrypt(nonce, plain, None)
        self.backup_path.write_bytes(self.salt + nonce + ct)
        self.backup_path.chmod(0o600)
        logging.debug("Backup file saved.")

    def _prune_old_keys(self) -> None:
        """Keep only the newest MAX_KEYS entries."""
        if len(self.keys) <= MAX_KEYS:
            return
        # key_ids sort lexicographically by timestamp string
        sorted_ids = sorted(self.keys.keys())
        for old_id in sorted_ids[: len(self.keys) - MAX_KEYS]:
            del self.keys[old_id]
            logging.info(f"Pruned old key: {old_id}")

    def _rotate_key(self, first: bool = False) -> None:
        new_key = os.urandom(32)
        if not first and self.current_id:
            prev = self.keys[self.current_id]
            # blend with previous to avoid sudden orphan
            new_key = bytes(a ^ b for a, b in zip(new_key, prev))
        key_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.keys[key_id] = new_key
        self.current_id = key_id
        self.usage_count = 0
        self.last_rotation = datetime.utcnow()
        self._prune_old_keys()
        logging.info(f"Rotated key; new current_id={self.current_id}")
        self._save_backup()

    def maybe_rotate(self) -> None:
        """Check thresholds and rotate if needed (thread-safe)."""
        with self._lock:
            now = datetime.utcnow()
            if (
                self.usage_count >= USAGE_ROTATE_THRESHOLD
                or (now - self.last_rotation) >= timedelta(hours=TIME_ROTATE_THRESHOLD_HOURS)
            ):
                self._rotate_key()
            else:
                # no rotation: just persist updated usage_count
                self._save_backup()

    def get_current_key(self) -> Tuple[str, bytes]:
        return self.current_id, self.keys[self.current_id]

    def get_key(self, key_id: str) -> Optional[bytes]:
        return self.keys.get(key_id)


key_manager = KeyManager()

# ─── Encryption / Decryption API ───────────────────────────────────────────────

def encrypt_txt(plaintext: str) -> str:
    key_id, key = key_manager.get_current_key()
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext.encode(), None)
    payload = {
        "key_id": key_id,
        "nonce": base64.b64encode(nonce).decode(),
        "ct":    base64.b64encode(ct).decode(),
    }
    key_manager.usage_count += 1
    key_manager.maybe_rotate()
    return base64.b64encode(json.dumps(payload).encode()).decode()

def decrypt_txt(token: str) -> str:
    payload = json.loads(base64.b64decode(token).decode())
    key = key_manager.get_key(payload["key_id"])
    if key is None:
        raise ValueError(f"Key not found: {payload['key_id']}")
    aes = AESGCM(key)
    nonce = base64.b64decode(payload["nonce"])
    ct    = base64.b64decode(payload["ct"])
    return aes.decrypt(nonce, ct, None).decode()

# ─── Coinbase OHLC Fetcher ────────────────────────────────────────────────────

async def fetch_ohlc(
    product_id: str = "ETH-USD",
    gran_sec: int = 4 * 3600,
    timeout: float = 10.0,
) -> pd.DataFrame:
    now = datetime.utcnow()
    start = now - timedelta(seconds=gran_sec * 50)
    params = {
        "start":       start.isoformat(),
        "end":         now.isoformat(),
        "granularity": gran_sec,
    }
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time")

# ─── GPT-4o Vision Debate Prompt Builder ─────────────────────────────────────

def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def build_hypertime_debate_prompts(chart_b64, market_context):
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
async def gpt4o_vision_debate_analysis(image_b64: str, market_context: str) -> str:
    client = OpenAI(api_key=OPENAI_KEY)
    messages = build_hypertime_debate_prompts(image_b64, market_context)
    resp = await client.chat.completions.create_async(
        model="gpt-4o", messages=messages, max_tokens=32000
    )
    return resp.choices[0].message.content

# ─── Dashboard / UI ───────────────────────────────────────────────────────────

class AdvancedDashboard(ctk.CTk):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self.title("GPT-4o Hypertime Mesh Consensus")
        self.geometry("1280x1100")
        self.configure(padx=20, pady=20)

        # Controls
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", pady=(0, 10))
        self.symbol_var = ctk.StringVar(value=COINBASE_SYMBOLS[0])
        self.gran_var = ctk.StringVar(value="4h")

        self.symbol_menu = ctk.CTkOptionMenu(ctrl, variable=self.symbol_var, values=COINBASE_SYMBOLS)
        self.symbol_menu.grid(row=0, column=0, padx=5)
        self.gran_menu = ctk.CTkOptionMenu(ctrl, variable=self.gran_var, values=["4h", "5h"])
        self.gran_menu.grid(row=0, column=1, padx=5)

        self.refresh_btn = ctk.CTkButton(
            ctrl, text="Refresh Now", command=lambda: self.schedule_update(True)
        )
        self.refresh_btn.grid(row=0, column=2, padx=5)

        # Chart & Analysis display
        self.chart_label = ctk.CTkLabel(self, text="")
        self.chart_label.pack(pady=10)
        self.consensus_text = ctk.CTkTextbox(self, width=1200, height=460, font=("Consolas", 13), wrap="word")
        self.consensus_text.configure(border_color="#444444", fg_color="#222222")
        self.consensus_text.pack(pady=10)

        # Status bar
        self.status = ctk.CTkLabel(self, text="Ready", font=("Arial", 12), text_color="gray")
        self.status.pack(pady=5)

        # Kick off periodic updates
        self.after(1000, lambda: self.schedule_update(False))
        self.after(180_000, lambda: self.schedule_update(False))  # every 3 min

    def schedule_update(self, force: bool = False) -> None:
        """Threadsafe scheduling of the async update."""
        self.status.configure(text="Scheduling update...")
        coro = self._update(force)
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        fut.add_done_callback(self._on_update_done)

    def _on_update_done(self, fut: asyncio.Future) -> None:
        err = fut.exception()
        if err:
            logging.exception("Error during update", exc_info=err)
            self.consensus_text.delete("0.0", "end")
            self.consensus_text.insert("0.0", f"Error: {err}")

    async def _update(self, force: bool) -> None:
        """Fetch data, render chart, run debate analysis."""
        self.status.configure(text="Fetching OHLC from Coinbase…")
        pid = self.symbol_var.get()
        gran = self.gran_var.get()
        gran_sec = 4 * 3600 if gran == "4h" else 5 * 3600

        try:
            df = await fetch_ohlc(product_id=pid, gran_sec=gran_sec)
        except Exception as e:
            self.status.configure(text="Fetch error")
            raise

        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        # Build Plotly figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="yellow",
                decreasing_line_color="white",
                increasing_line_width=1.5,
                decreasing_line_width=1.5,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["ema20"],
                name="EMA20",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )
        fig.update_layout(
            title=dict(text=f"{pid} {gran.upper()} Candles", x=0.5, xanchor="center"),
            template="plotly_dark",
            height=600,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.2, title="Time"),
            yaxis=dict(showgrid=True, gridcolor="gray", gridwidth=0.2, title="Price (USD)"),
        )

        img = to_image(fig, format="png")
        b64 = image_to_base64(img)
        photo = tk.PhotoImage(data=b64, format="png")
        chart_tk = ctk.CTkImage(light_image=photo, size=(1200, 600))
        self.chart_label.configure(image=chart_tk)
        self.chart_label.image = chart_tk

        last = df["close"].iloc[-1]
        context = (
            f"{pid} = ${last:.2f}. Range: {df['low'].min():.2f}–{df['high'].max():.2f} {gran.upper()}. "
            "Use hypertime error-correction and debate scenario failures."
        )

        self.status.configure(text="Running hypertime mesh debate…")
        analysis = await gpt4o_vision_debate_analysis(b64, context)

        # Extract consensus
        if "[replytemplate]" in analysis and "[/replytemplate]" in analysis:
            a = analysis.index("[replytemplate]") + len("[replytemplate]")
            b = analysis.index("[/replytemplate]")
            mesh = analysis[a:b].strip()
        else:
            mesh = (analysis[:9000] + "\n...\n[Truncated]") if len(analysis) > 9000 else analysis

        self.consensus_text.delete("0.0", "end")
        self.consensus_text.insert("0.0", mesh)
        self.status.configure(text="Analysis complete.")

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Start asyncio loop in background thread
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    app = AdvancedDashboard(loop)
    app.mainloop()

if __name__ == "__main__":
    main()
