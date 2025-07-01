# Relational Philosophy Simulation

A prototype Agent-Based Model (ABM) of Amazon hydropower governance that embeds **relational philosophy** (“hydrosocial territories”) into a Mesa simulation, coupled with an Ollama-backed LLM (DeepSeekModel) to generate real-time narrative feedback and adaptive compensation policies.

---

## 🔍 Features

- **Heterogeneous stakeholders**: operators, communities, regulators  
- **Environmental agent**: dynamic reservoir-level decay  
- **Judging agent**: aggregates votes, computes consensus & conflict  
- **LLM-driven narratives**: one-line summaries & policy recommendations  
- **Adaptive compensation**: adjusts FCUWR (compensation) multiplier on the fly  
- **Latin-Hypercube exploration**: sweeps 7 key axes over 20 scenarios  
- **Data logging**: combined CSV outputs for downstream analysis  

---

## 🚀 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/cpscesar/hydrosocial-abm-llm.git
   cd hydrosocial-abm-llm
   ```

2. **Install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ollama pull deepseek-r1:70b
   ```

> _Requires Python 3.10+; Ollama CLI installed and the `deepseek-r1:70b` model pulled locally._

---

## ⚙️ Usage

Run the main simulation script:

```bash
python run.py
```

By default it will:

1. Sample 20 scenarios via Latin-Hypercube  
2. For each scenario: run 1 replicate of 10 time steps  
3. Log **agent-level** and **model-level** metrics to `combined_agent_data.csv`  
4. Log **narrative feedback** to `combined_narrative_data.csv`  

You can edit at the bottom of `run.py`:

- `N` (number of LHS samples)  
- `runs_per_scenario` and `steps_per_scenario`  
- Grid `width` and `height`  

---

## 🔧 Scenario parameters

Each entry in `scenario_params` controls one axis of your experiment:

| Parameter                 | Type    | Description                                                      |
|---------------------------|---------|------------------------------------------------------------------|
| `initial_water`           | float   | Starting reservoir level (0–100)                                 |
| `decay_rate`              | float   | Max water drop per step                                          |
| `compensation_adjustment` | float   | Initial FCUWR multiplier                                         |
| `num_communities`         | int     | Number of community agents (e.g. 1, 3, 5)                        |
| `power_asymmetry`         | float   | Operator’s technical advantage factor                            |
| `conflict_threshold`      | float   | Water-level below which conflict intensity starts accumulating   |
| `llm_mode`                | string  | Narrative mode: `off`, `baseline`, or `aggressive`               |

---

## 📊 Outputs

- **`combined_agent_data.csv`**  
  Contains per-step, per-agent raw & ordinal metrics, decisions & conflict.

- **`combined_narrative_data.csv`**  
  Contains per-step narrative summaries and updated compensation multiplier.

Use your preferred analysis notebook or tool (e.g., pandas, Jupyter, Power BI) to visualize time series, conflict trends, and word clouds of narrative feedback.

---

## 🛠️ Next steps

- **Empirical calibration** against field data  
- **Spatial extension** with realistic river reaches  
- **Participatory narrative prompts** co-designed with local stakeholders  
- **CLI flags** or config file support for scenario axes  

---

## 📄 License & Citation

If you use this work, please cite:

> Soares, C.P. (2025). *Relational Philosophy Smulation*. GitHub repository, https://github.com/cpscesar/hydrosocial-abm-llm
