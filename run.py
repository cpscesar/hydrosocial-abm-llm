#!/usr/bin/env python
"""
Hydropower ABM with Integrated LLM for Testing Waterscape and Hydrosocial Theories

This simulation models interactions among heterogeneous stakeholders in a hydropower context.
It integrates:
  1. A Mesa-based ABM simulation engine representing hydropower operators, communities, regulators,
     and environmental agents (e.g., a water agent).
  2. An LLM module (via DeepSeekModel through Ollama) that analyzes simulation outputs,
     generates narrative summaries, and proposes policy feedback.
  3. A data interface that collects quantitative metrics (resource allocation, conflict frequency,
     compensation indices, socioenvironmental impact, etc.) and qualitative narrative outputs
     for visualization and analysis.

Scenario parameters (in `scenario_params` dict):
  • initial_water (float): starting reservoir level (%)
  • decay_rate (float): maximum water‐level drop per step
  • compensation_adjustment (float): initial FCUWR multiplier
  • num_communities (int): number of community agents (1, 3, or 5)
  • power_asymmetry (float): operator’s technical advantage factor
  • conflict_threshold (float): water‐level threshold triggering conflict
  • llm_mode (str): narrative feedback mode; one of “off”, “baseline”, “aggressive”

Key Outputs:
  - Quantitative: Water usage, energy production, water flow, conflict intensity, compensation index,
    socioenvironmental impact, spatial agent positions, and temporal trends.
  - Qualitative: Narrative summaries, interpretive reports, hypothesis generation, and policy feedback.
  
This version supports hypothesis-driven experiments. All agents update their values dynamically,
each agent has an explicit objective definition so the LLM can judge the situation appropriately,
and a narrative is generated at every simulation step.

Note: All metric values are expressed as ordinal categories on a scale from 1 (lowest) to 5 (highest)
based on the following ranges:
      Water Usage: 5–15, Energy Production: 20–50, Water Flow: 10–30, Water Level: 0–100, 
      Compensation Index: 0–2, Socioenvironmental Impact: 0–2.
"""

import os
import random
import time
import re
import subprocess
import csv
import json
import numpy as np
from tqdm import tqdm  # progress bar
from itertools import product

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from pydantic import Field, BaseModel
from langchain.llms.base import LLM

# ===========================
# 0. Helper: Ordinal Conversion Function
# ===========================
def to_ordinal(metric_name, value):
    """Converts a numeric value to an ordinal category (1-5) based on predefined ranges.
    Clamps the value so that it does not exceed the min/max defined."""
    ranges = {
        "water_usage": (5, 15),
        "energy_production": (20, 50),
        "water_flow": (10, 30),
        "water_level": (0, 100),
        "compensation_index": (0, 2),
        "socioenviroimpact": (0, 2)
    }
    min_val, max_val = ranges.get(metric_name, (0, 1))
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    normalized = (value - min_val) / (max_val - min_val)
    category = int(normalized * 4) + 1
    if category < 1:
        category = 1
    if category > 5:
        category = 5
    return category

# ===========================
# 1. LLM Module (Language Processing)
# ===========================
def remove_thinking(text: str) -> str:
    """Remove any content enclosed in <think>...</think> tags."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

class DeepSeekModel(LLM, BaseModel):
    model_name: str = Field(default="deepseek-r1:70b")
    def _call(self, prompt: str, stop=None) -> str:
        command = ['ollama', 'run', self.model_name]
        try:
            result = subprocess.run(command, input=prompt, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            output = remove_thinking(output)
            if stop:
                for token in stop:
                    output = output.split(token)[0]
            return output
        except subprocess.CalledProcessError as e:
            raise Exception(f"Ollama command failed: {e.stderr}")
    @property
    def _llm_type(self):
        return "deepseek"

class CleanerLLM(LLM, BaseModel):
    model_name: str = Field(default="deepseek-r1:70b")
    def _call(self, prompt, stop=None, use_evac=False):
        command = ['ollama', 'run', self.model_name]
        try:
            result = subprocess.run(command, input=prompt, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            if stop:
                for token in stop:
                    output = output.split(token)[0]
            return output
        except subprocess.CalledProcessError as e:
            raise Exception(f"Cleaner command failed: {e.stderr}")
    @property
    def _llm_type(self):
        return "cleaner"
    def clean(self, raw_text: str, format_instruction: str, use_evac=False) -> str:
        cleaning_prompt = (
            f"Raw LLM response: {raw_text}\n"
            f"Extract and output exactly one line following this format: {format_instruction}\n"
            "Output exactly in that format, with no additional text."
        )
        cleaned = self._call(cleaning_prompt, use_evac=use_evac)
        return cleaned.strip()

# Global LLM instances
evac_llm = DeepSeekModel()
other_llm = DeepSeekModel()
evac_cleaner_llm = CleanerLLM(model_name="deepseek-r1:70b")
other_cleaner_llm = CleanerLLM(model_name="deepseek-r1:70b")

def call_llm(prompt, expected_format=None, use_evac=False):
    chosen_llm = evac_llm if use_evac else other_llm
    print(f"[LLM] Using model: {chosen_llm.model_name} (use_evac={use_evac})")
    max_attempts = 5
    response = ""
    for attempt in range(max_attempts):
        response = chosen_llm._call(prompt)
        response = response.strip()
        if response:
            break
    if not response:
        response = "No answer"
    response = re.sub(r'\s+', ' ', response)
    if expected_format:
        chosen_cleaner = evac_cleaner_llm if use_evac else other_cleaner_llm
        response = chosen_cleaner.clean(response, expected_format, use_evac=use_evac)
    return response

# ===========================
# 2. Data Interface Functions
# ===========================
def save_csv(data, filename, fieldnames, write_header=False):
    mode = "a"  # Always append to the combined CSV
    file_exists = os.path.exists(filename)
    try:
        with open(filename, mode, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in data:
                writer.writerow(row)
        print(f"[DATA] Saved data to {filename}")
    except Exception as e:
        print(f"[DATA] Error saving CSV: {e}")

# ===========================
# 3. Utility Functions
# ===========================
def generate_names(n: int):
    fallback_names = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Sam", "Jamie", "Dana", "Lee"]
    return [fallback_names[i % len(fallback_names)] for i in range(n)]

# ===========================
# 4. ABM Agents (Simulation Engine)
# ===========================
class StakeholderAgent(Agent):
    """
    Represents a stakeholder.

    Objectives:
      - Operator: Represents a hydropower operator focused on maximizing energy production and technical efficiency.
      - Community: Represents indigenous and traditional communities that prioritize environmental sustainability, cultural heritage, and equitable compensation.
      - Regulator: Represents government regulatory bodies ensuring environmental compliance, public safety, and fair compensation policies.

    Quantitative metrics (expressed as ordinal categories on a scale of 1 to 5):
      - Water usage, energy production, water flow, water level, compensation index, socioenvironmental impact.
      
    The agent uses an LLM call to decide whether to implement a compensation policy.
    Raw metric values are preserved in the CSV alongside ordinal categories.
    """
    def __init__(self, unique_id, model, full_name, group=None):
        super().__init__(unique_id, model)
        self.full_name = full_name
        if group is None:
            self.group = random.choice(["operator", "community", "regulator"])
        else:
            self.group = group
        if self.group == "operator":
            self.objective = "Represents a hydropower operator focused on maximizing energy production and technical efficiency."
        elif self.group == "community":
            self.objective = "Represents indigenous and traditional communities that prioritize environmental sustainability, cultural heritage, and equitable compensation."
        elif self.group == "regulator":
            self.objective = "Represents government regulatory bodies ensuring environmental compliance, public safety, and fair compensation policies."
        
        # Initialize raw metrics.
        self.water_usage = random.uniform(5, 15)
        self.energy_production = random.uniform(20, 50)
        self.water_flow = random.uniform(10, 30)
        self.conflict_intensity = 0.0
        self.compensation_index = 0.0
        self.socioenviroimpact = 0.0
        self.last_decision = ""
        self.last_reason = ""
    
    def step(self):
        global_water = self.model.water_level
        # Update raw values based on current water level.
        if self.group == "operator":
            # give the operator a mechanical advantage
            self.energy_production *= self.model.power_asymmetry
            self.water_usage = self.water_usage * (global_water / 100)
            self.energy_production = max(0, self.energy_production * (global_water / 100))
            self.water_flow = self.water_flow * (global_water / 100)
            self.socioenviroimpact = 0.2 * ((100 - global_water) / 100)
        elif self.group == "community":
            self.water_usage = self.water_usage * (1 + (100 - global_water) / 200)
            self.energy_production = self.energy_production * (global_water / 100)
            self.water_flow = self.water_flow * (global_water / 100)
            self.socioenviroimpact = (100 - global_water) / 100 + self.conflict_intensity / 10
        elif self.group == "regulator":
            self.water_usage = self.water_usage * (1 + (100 - global_water) / 400)
            self.energy_production = self.energy_production * (global_water / 100)
            self.water_flow = self.water_flow * (global_water / 100)
            self.socioenviroimpact = self.conflict_intensity / 20
        
        self.compensation_index = (global_water / 100.0) * self.model.compensation_adjustment
        
        # Convert raw values to ordinal categories.
        water_level_ord = to_ordinal("water_level", global_water)
        water_usage_ord = to_ordinal("water_usage", self.water_usage)
        energy_prod_ord = to_ordinal("energy_production", self.energy_production)
        water_flow_ord = to_ordinal("water_flow", self.water_flow)
        comp_index_ord = to_ordinal("compensation_index", self.compensation_index)
        socioenviro_ord = to_ordinal("socioenviroimpact", self.socioenviroimpact)
        
        # Build prompt with ordinal values and a note about the ordinal scale.
        prompt = (
            f"Note: All metric values are expressed as ordinal categories on a scale from 1 (lowest) to 5 (highest), "
            f"based on these ranges: Water Usage (5-15), Energy Production (20-50), Water Flow (10-30), Water Level (0-100), "
            f"Compensation Index (0-2), Socioenvironmental Impact (0-2). "
            f"Agent {self.full_name} ({self.group}) has the objective: {self.objective}. "
            f"Current ordinal metrics - Water level: {water_level_ord}, Water usage: {water_usage_ord}, Energy production: {energy_prod_ord}, "
            f"Water flow: {water_flow_ord}, Compensation index: {comp_index_ord}, Socioenvironmental impact: {socioenviro_ord}. "
            f"Based on your objective and these indicators, decide whether to implement the policy. "
            f"Respond in exactly one line in the format: 'Response: <implement or not_implement>. Reason: <brief explanation>'."
        )
        
        pattern = r"Response:\s*(implement|not_implement)[\.\:]?\s*Reason:\s*(.+)"

        # response = call_llm(prompt, expected_format="Response: <implement or not_implement>. Reason: <brief explanation>", use_evac=True)
        if self.model.llm_mode == "off":
            # no LLM → default decision
            decision, reason = "not_implement", "LLM disabled"
        else:
            response = call_llm(prompt,
                                expected_format="Response: <implement or not_implement>. Reason: <brief explanation>",
                                use_evac=True)
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                decision, reason = match.group(1).lower(), match.group(2).strip()
            else:
                decision, reason = "not_implement", "No clear response"
        
        self.last_decision = decision
        self.last_reason = reason
        
        # Update conflict intensity continuously when water level is below 60.
        threshold = self.model.conflict_threshold
        if global_water < threshold:  
            # base_conflict = (60 - global_water) / 10  # proportional increment
            base_conflict = (self.model.conflict_threshold - global_water) / 10
            if (self.group == "community" and decision == "not_implement") or (self.group == "operator" and decision == "implement"):
                self.conflict_intensity += base_conflict
        
        print(f"[STAKEHOLDER {self.full_name} | {self.group}] Objective: {self.objective} | Decision: {decision}, Reason: {reason}, Conflict: {self.conflict_intensity:.2f}, SocioEnviroImpact: {self.socioenviroimpact:.2f}")

class JudgeAgent(Agent):
    """
    Aggregates stakeholder decisions and overall conflict intensity to determine consensus.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        decisions = [agent.last_decision for agent in self.model.schedule.agents if isinstance(agent, StakeholderAgent)]
        count_impl = decisions.count("implement")
        count_not = decisions.count("not_implement")
        total = len(decisions)
        if total > 0:
            if count_impl > total / 2:
                self.model.judge_decision = "Consensus: implement"
            elif count_not > total / 2:
                self.model.judge_decision = "Consensus: not_implement"
            else:
                self.model.judge_decision = "No consensus"
        self.model.total_conflict = sum(agent.conflict_intensity for agent in self.model.schedule.agents if isinstance(agent, StakeholderAgent))
        print(f"[JUDGE] {self.model.judge_decision} | Total Conflict: {self.model.total_conflict:.2f}")

class WaterAgent(Agent):
    """
    Simulates changes in water level, a key environmental indicator.
    """
    def __init__(self, unique_id, model, initial_level=100.0):
        super().__init__(unique_id, model)
        self.level = initial_level
    def step(self):
        self.level -= random.uniform(0, self.model.decay_rate)
        self.model.water_level = self.level
        if self.level < 50:
            print(f"[WATER] Alert: Water level low at {self.level:.1f}!")
        else:
            print(f"[WATER] Water level: {self.level:.1f}")

# ===========================
# 5. Central Simulation Hub (ABM Model)
# ===========================
class HydropowerModel(Model):
    """
    The central simulation hub that orchestrates agents, logs detailed metrics,
    and interacts with the LLM module to generate qualitative narrative outputs and policy recommendations.

    Quantitative Outputs:
      - Agent metrics: water usage, energy production, water flow, conflict intensity, compensation index, socioenvironmental impact.
      - Environmental metrics: water level, total conflict.
      - Spatial data: agent positions (via grid).
      - Temporal trends: captured via the day_counter.

    Qualitative Outputs:
      - Narrative summaries, interpretive reports, hypothesis generation, and policy recommendations generated by the LLM using aggregated simulation state.

    Supports hypothesis-driven experiments by adjusting parameters based on LLM feedback.
    """
    def __init__(self, width, height, scenario_params):
        # scenario_params is a dictionary with keys: initial_water, decay_rate, compensation_adjustment.
        # self.num_agents = 3
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.day_counter = 0
        self.judge_decision = ""
        self.water_level = scenario_params.get("initial_water", 100.0)
        self.decay_rate = scenario_params.get("decay_rate", 2.0)
        self.narrative_feedback = ""
        self.total_conflict = 0
        self.compensation_adjustment = scenario_params.get("compensation_adjustment", 1.0)

        self.initial_water       = scenario_params["initial_water"]
        self.decay_rate_setting  = scenario_params["decay_rate"]
        self.comp_adj_setting    = scenario_params["compensation_adjustment"]
        self.num_communities     = scenario_params["num_communities"]
        self.power_asymmetry     = scenario_params["power_asymmetry"]
        self.conflict_threshold  = scenario_params["conflict_threshold"]
        self.llm_mode            = scenario_params["llm_mode"]
        
        # ── WIRE new axes ───────────────────────────
        # self.num_communities    = scenario_params.get("num_communities", 1)
        # self.power_asymmetry    = scenario_params.get("power_asymmetry", 1.0)
        # self.conflict_threshold = scenario_params.get("conflict_threshold", 90)

        # ── WIRE llm_mode ───────────────────
        # self.llm_mode = scenario_params.get("llm_mode", "baseline")

        # groups = ["operator", "community", "regulator"]
        # 1 operator, N communities, 1 regulator
        groups = ["operator"] + ["community"] * self.num_communities + ["regulator"]
        names  = generate_names(len(groups))

        for i, group in enumerate(groups):
            agent = StakeholderAgent(unique_id=i, model=self, full_name=names[i], group=group)
            self.schedule.add(agent)
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(agent, (x, y))
        
        base_id = len(groups)
        judge = JudgeAgent(unique_id=base_id, model=self)
        water = WaterAgent(unique_id=base_id+1, model=self, initial_level=self.water_level)
        self.schedule.add(judge)
        
        self.schedule.add(water)
        
        self.datacollector = DataCollector(
            model_reporters={
                "Initial_Water":       lambda m: m.initial_water,
                "Decay_Rate":          lambda m: m.decay_rate_setting,
                "Comp_Adj_Setting":    lambda m: m.comp_adj_setting,
                "Num_Communities":     lambda m: m.num_communities,
                "Power_Asymmetry":     lambda m: m.power_asymmetry,
                "Conflict_Threshold":  lambda m: m.conflict_threshold,
                "LLM_Mode":            lambda m: m.llm_mode,
                "Water_Level": lambda m: m.water_level,
                "Judge_Decision": lambda m: m.judge_decision,
                "Total_Conflict": lambda m: m.total_conflict,
                "Compensation_Index_Avg": lambda m: np.mean([agent.compensation_index for agent in m.schedule.agents if isinstance(agent, StakeholderAgent)]),
                "Compensation_Adjustment": lambda m: m.compensation_adjustment
            },
            agent_reporters={
                "Group": lambda a: getattr(a, "group", None),
                "Water_Usage": lambda a: getattr(a, "water_usage", None),
                "Energy_Production": lambda a: getattr(a, "energy_production", None),
                "Water_Flow": lambda a: getattr(a, "water_flow", None),
                "Conflict_Intensity": lambda a: getattr(a, "conflict_intensity", None),
                "Decision": lambda a: getattr(a, "last_decision", None),
                "Reason": lambda a: getattr(a, "last_reason", None),
                "SocioEnviroImpact": lambda a: getattr(a, "socioenviroimpact", None)
            }
        )
    
    def generate_simulation_summary(self):
        summary = f"Day {self.day_counter}: Water level is {self.water_level:.1f}. "
        stakeholder_info = []
        for agent in self.schedule.agents:
            if isinstance(agent, StakeholderAgent):
                water_usage_ord = to_ordinal("water_usage", agent.water_usage)
                energy_prod_ord = to_ordinal("energy_production", agent.energy_production)
                water_flow_ord = to_ordinal("water_flow", agent.water_flow)
                comp_index_ord = to_ordinal("compensation_index", agent.compensation_index)
                socioenviro_ord = to_ordinal("socioenviroimpact", agent.socioenviroimpact)
                stakeholder_info.append(
                    f"{agent.full_name} ({agent.group}): Decision {agent.last_decision}, " +
                    f"Usage {water_usage_ord}, Energy {energy_prod_ord}, Flow {water_flow_ord}, Impact {socioenviro_ord}"
                )
        summary += "Stakeholder metrics (ordinal): " + " | ".join(stakeholder_info) + ". "
        summary += f"Total conflict intensity: {self.total_conflict:.2f}. "
        avg_comp = np.mean([agent.compensation_index for agent in self.schedule.agents if isinstance(agent, StakeholderAgent)])
        summary += f"Average compensation index: {avg_comp:.2f}. "
        summary += "Judge decision: " + self.judge_decision + ". "
        summary += f"Policy compensation adjustment factor: {self.compensation_adjustment:.2f}."
        print(f"[SUMMARY] {summary}")
        
        prompt = (
            f"Note: All metric values in this summary are expressed as ordinal categories on a scale from 1 (lowest) to 5 (highest) "
            f"based on these ranges: Water Usage: 5-15, Energy Production: 20-50, Water Flow: 10-30, Water Level: 0-100, "
            f"Compensation Index: 0-2, Socioenvironmental Impact: 0-2. "
            f"Summarize the simulation state and emergent trends using the following quantitative data:\n"
            f"{summary}\n\n"
            "Provide an interpretive narrative that explains how water resources are managed, how social conflicts emerge, "
            "and propose policy adjustments. Specifically, generate a hypothesis about why conflicts are occurring and "
            "suggest whether compensation policies should be increased or reduced. "
            "Respond in exactly one line in the format: 'Narrative: <narrative summary>. Recommendation: <policy recommendation>'."
        )
        narrative = call_llm(prompt, expected_format="Narrative: <narrative summary>. Recommendation: <policy recommendation>", use_evac=True)
        if not narrative or narrative.lower() == "no answer":
            narrative = "Narrative: No significant emergent trends observed. Recommendation: No policy change."
        self.narrative_feedback = narrative
        print(f"[NARRATIVE] {narrative}")
        self.adjust_parameters_from_feedback()
    
    def adjust_parameters_from_feedback(self):
        feedback = self.narrative_feedback.lower()
        if "increase compensation" in feedback:
            factor = 1.1
        elif "reduce compensation" in feedback:
            factor = 0.9
        else:
            return

        if self.llm_mode == "aggressive":
            # square the factor to amplify changes
            factor = factor ** 2

        self.compensation_adjustment *= factor
        print(f"[ADJUST] Compensation adjustment factor is now {self.compensation_adjustment:.2f}")
    
    def step(self):
        self.schedule.step()
        self.day_counter += 1 
        self.datacollector.collect(self)
        print(f"[MODEL] Completed step {self.day_counter}")
        self.generate_simulation_summary()

# ===========================
# 6. Main Simulation Loop
# ===========================
if __name__ == '__main__':
    # ───── 1. Define your axes ──────────────────────
    axis_names = [
        "initial_water",
        "decay_rate",
        "compensation_adjustment",
        "num_communities",
        "power_asymmetry",
        "conflict_threshold",
        "llm_mode",
    ]
    axis_values = [
        [10.0, 50.0, 90.0, 100.0],           # initial_water
        [0.5, 1.0, 2.0, 3.0],                 # decay_rate
        [0.5, 0.8, 1.0, 1.2, 1.5],            # compensation_adjustment
        [1, 3, 5],                            # num_communities
        [0.5, 1.0, 2.0],                      # power_asymmetry
        [60, 80, 100],                        # conflict_threshold
        ["off", "baseline", "aggressive"],    # llm_mode
    ]

    # ───── 2. Latin-Hypercube sampler ───────────────
    def lhs_sample(n_samples, n_dims):
        """Return an (n_samples × n_dims) LHS in [0,1)."""
        result = np.zeros((n_samples, n_dims))
        for dim in range(n_dims):
            perm = np.random.permutation(n_samples)
            result[:, dim] = (perm + np.random.rand(n_samples)) / n_samples
        return result

    # ───── 3. Draw 100 LHS points ──────────────────
    np.random.seed(42)   # reproducible
    N = 20
    raw = lhs_sample(N, len(axis_values))

    # ───── 4. Map to discrete levels ───────────────
    scenarios = []
    for row in raw:
        scen = {}
        for dim, x in enumerate(row):
            levels = axis_values[dim]
            idx = min(int(x * len(levels)), len(levels) - 1)
            scen[axis_names[dim]] = levels[idx]
        scenarios.append(scen)

    print(f"Built {len(scenarios)} LHS scenarios")

    # ───── 5. Run each scenario ────────────────────
    runs_per_scenario = 1
    steps_per_scenario = 10
    print(f"→ {len(scenarios)} scenarios × {runs_per_scenario} run × "
      f"{steps_per_scenario} steps = "
      f"{len(scenarios) * runs_per_scenario * steps_per_scenario} total model.step() calls")

    combined_agent_csv     = "combined_agent_data.csv"
    combined_narrative_csv = "combined_narrative_data.csv"

    # remove old files
    for fn in [combined_agent_csv, combined_narrative_csv]:
        if os.path.exists(fn):
            os.remove(fn)

    # fieldnames stay the same as before
    agent_fieldnames = [
        "Scenario","Run","Step","Agent","Group",
        "Water_Usage","Energy_Production","Water_Flow",
        "Conflict_Intensity","Decision","Reason",
        "Compensation_Index","SocioEnviroImpact"
    ]
    narrative_fieldnames = [
        "Scenario","Run","Step","Narrative","Compensation_Adjustment"
    ]

    # your existing loops — just swap in `scenarios` from above
    for scenario in tqdm(scenarios, desc="Scenarios", unit="scenario"):
        scenario_label = (
            f"water={scenario['initial_water']},"
            f"decay={scenario['decay_rate']},"
            f"comp_adj={scenario['compensation_adjustment']},"
            f"comms={scenario['num_communities']},"
            f"power={scenario['power_asymmetry']},"
            f"th={scenario['conflict_threshold']},"
            f"llm={scenario['llm_mode']}"
        )
        for run in range(1, runs_per_scenario+1):
            print(f"\n=== Scenario: {scenario_label} (Run {run}) ===\n")
            model = HydropowerModel(width=10, height=10, scenario_params=scenario)
            for step in range(steps_per_scenario):
                model.step()
                # 1) Gather all agents’ data for this step
                agent_data = [
                    {
                        "Scenario": scenario_label,
                        "Run": run,
                        "Step": step + 1,
                        "Agent": agent.unique_id,
                        "Group": agent.group,
                        "Water_Usage": agent.water_usage,
                        "Energy_Production": agent.energy_production,
                        "Water_Flow": agent.water_flow,
                        "Conflict_Intensity": agent.conflict_intensity,
                        "Decision": agent.last_decision,
                        "Reason": agent.last_reason,
                        "Compensation_Index": agent.compensation_index,
                        "SocioEnviroImpact": agent.socioenviroimpact,
                    }
                    for agent in model.schedule.agents
                    if isinstance(agent, StakeholderAgent)
                ]
                save_csv(agent_data, combined_agent_csv, agent_fieldnames)

                # 2) Save the narrative summary for this step
                narrative_row = {
                    "Scenario": scenario_label,
                    "Run": run,
                    "Step": step + 1,
                    "Narrative": model.narrative_feedback,
                    "Compensation_Adjustment": model.compensation_adjustment,
                }
                save_csv([narrative_row], combined_narrative_csv, narrative_fieldnames)

    print("All LHS scenarios complete.")
