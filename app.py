import streamlit as st
import numpy as np
import pandas as pd
import os

# ✅ PERBAIKAN UTAMA:
from src.env_qlearning import PenjualanEnv
from enhanced_qlearning import EnhancedPenjualanEnv, QLearningAgent

# ==================================================
# ⚡️ 1. Load Model dan Data
# ==================================================
OUTPUT_DIR = "output"
q_table = np.load(f"{OUTPUT_DIR}/enhanced_q_table.npy")
state_idx = np.load(f"{OUTPUT_DIR}/state_idx.npy", allow_pickle=True).item()
baseline_profit = float(open(f"{OUTPUT_DIR}/baseline_profit.txt").read())

# Init environment dan agent
env = EnhancedPenjualanEnv("env_SAR.csv")
agent = QLearningAgent(len(env.states), len(env.actions))
agent.q_table = q_table

# ==================================================
# ⚡️ 2. Tampilan Streamlit
# ==================================================
st.title("💵 Simulasi Q-learning untuk Strategi Harga")
st.write("Pilih st
