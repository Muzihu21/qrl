import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning_env import PenjualanEnv

# ========== Setup ==========
st.set_page_config(page_title="Q-Learning Harga", layout="wide")

# ========== Load Environment ==========
env = PenjualanEnv()
env.unique_states = list(set(env.states))
env.n_states = len(env.unique_states)

# ========== Sidebar Menu ==========
menu = st.sidebar.radio("Pilih Halaman", [
    "📊 Visualisasi Q-table",
    "📈 Evaluasi Policy",
    "📉 Grafik Reward",
    "⚙️ Training Ulang",
    "ℹ️ Tentang"
])

# ========== Fungsi: Training ==========
def train_q_learning(env, alpha, gamma, epsilon, episodes):
    state_to_index = {s: i for i, s in enumerate(env.unique_states)}
    q_table = np.zeros((len(env.unique_states), env.n_actions))
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_idx = state_to_index[state]
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state_idx])

            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, done, _ = result

            next_state_idx = state_to_index[next_state]
            q_table[state_idx, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )

            total_reward += reward
            state = next_state
        rewards_per_episode.append(total_reward)

    return q_table, np.array(rewards_per_episode)

# ========== Fungsi: Evaluasi ==========
def evaluate_policy(env, q_table, n_trials=100):
    total_rewards = []
    for _ in range(n_trials):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            try:
                state_index = env.unique_states.index(state)
                action = np.argmax(q_table[state_index])
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, _ = result
                elif len(result) == 3:
                    next_state, reward, done = result
                else:
                    raise ValueError("env.step() return format tidak valid")

                episode_reward += reward
                state = next_state
            except Exception as e:
                st.warning(f"⚠️ Error evaluasi policy: {e}")
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# ========== Halaman: Heatmap ==========
if menu == "📊 Visualisasi Q-table":
    st.title("📊 Strategi Harga: Q-table Heatmap")
    try:
        q_table = np.load("q_table.npy")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(q_table, annot=True, cmap="YlGnBu",
                    xticklabels=env.harga_list,
                    yticklabels=env.unique_states,
                    ax=ax)
        ax.set_xlabel("Harga (Action)")
        ax.set_ylabel("State")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Gagal memuat Q-table: {e}")

# ========== Halaman: Evaluasi ==========
elif menu == "📈 Evaluasi Policy":
    st.title("📈 Evaluasi Policy")
    try:
        q_table = np.load("q_table.npy")
        trials = st.slider("Jumlah Simulasi Episode", 10, 10000, 100, step=100)
        avg_reward = evaluate_policy(env, q_table, trials)
        st.success(f"🎯 Rata-rata reward dari {trials} simulasi: **{avg_reward:.2f}**")
    except FileNotFoundError:
        st.error("❌ File `q_table.npy` tidak ditemukan.")

# ========== Halaman: Grafik Reward ==========
elif menu == "📉 Grafik Reward":
    st.title("📉 Grafik Reward per Episode")
    try:
        rewards = np.load("rewards_per_episode.npy")
        fig, ax = plt.subplots()
        ax.plot(rewards, label='Reward per Episode', color='green')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward per Episode (Training Progress)")
        ax.legend()
        st.pyplot(fig)
    except FileNotFoundError:
        st.error("❌ File `rewards_per_episode.npy` tidak ditemukan.")

# ========== Halaman: Training Ulang ==========
elif menu == "⚙️ Training Ulang":
    st.title("⚙️ Training Ulang Q-Learning")

    alpha = st.number_input("Alpha (Learning rate)", 0.0, 1.0, 0.1, step=0.01)
    gamma = st.number_input("Gamma (Discount factor)", 0.0, 1.0, 0.9, step=0.01)
    epsilon = st.number_input("Epsilon (Exploration rate)", 0.0, 1.0, 0.1, step=0.01)
    episodes = st.number_input("Jumlah Episode", 100, 10000, 1000, step=100)

    if st.button("🚀 Mulai Training"):
        with st.spinner("Training sedang berjalan..."):
            q_table, rewards = train_q_learning(env, alpha, gamma, epsilon, episodes)
            np.save("q_table.npy", q_table)
            np.save("rewards_per_episode.npy", rewards)
            st.success("✅ Training selesai dan file disimpan.")

# ========== Halaman: Tentang ==========
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari skripsi untuk mensimulasikan **Reinforcement Learning (Q-Learning)** 
    dalam konteks **penetapan harga produk**.

    **Fitur:**
    - Visualisasi Q-table (heatmap)
    - Evaluasi policy
    - Grafik reward per episode
    - Training ulang dengan hyperparameter custom

    **Author**: Zihu — AI Engineer & Pejuang Skripsi 🧠🔥  
    **Stack**: Python, Streamlit, NumPy, Matplotlib, Seaborn
    """)

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 — Made with ❤️ by Zihu | Powered by Streamlit")
