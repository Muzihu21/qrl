import streamlit as st
import numpy as np
import pandas as pd
import os

# ✅ PERBAIKAN UTAMA:
from src.env_qlearning import PenjualanEnv
from src.env_qlearning import EnhancedPenjualanEnv, QLearningAgent

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
st.write("Pilih state awal dan dapatkan saran aksi dari model.")

# Dropdown untuk pilih state awal
state_choice = st.selectbox("Pilih State Awal:", list(state_idx.keys()))
state_index = state_idx[state_choice]

# Aksi yang disarankan model
action_index = np.argmax(agent.q_table[state_index])
recommended_action = env.actions[action_index]
st.markdown(f"### 👉 Aksi Terbaik: **{recommended_action}**")

# Simulasi langkah selanjutnya
if st.button("Simulasi 1 Langkah"):
    next_state, reward, done = env.step(recommended_action)
    st.write(f"State Berikutnya: {next_state}")
    st.write(f"Reward: {reward:.2f}")
    st.write(f"Done: {done}")

# Info Tambahan
st.write("---")
st.write("Original Profit Mean:", baseline_profit)
