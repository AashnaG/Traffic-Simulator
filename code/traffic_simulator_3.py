import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import random
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightState(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    ALL_RED = 3

@dataclass
class Vehicle:
    id: int
    position: Tuple[float, float]
    destination: Tuple[float, float]
    speed: float
    max_speed: float = 10.0
    wait_time: float = 0.0
    current_road: Optional['Road'] = None
    next_intersection: Optional['Intersection'] = None
    lane_offset: float = 0.0

@dataclass
class Road:
    name: str
    length: float
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    queue: List['Vehicle'] = field(default_factory=list)
    queue_len: float = 0.0
    approach_speed: float = 10.0
    grade: float = 0.0
    # Added for history tracking for predictor
    history_queue: List[float] = field(default_factory=list)

    def stop_line(self) -> float:
        return max(0.0, self.length - 5.0)

@dataclass
class Intersection:
    id: int
    position: Tuple[float, float]
    incoming_roads: Dict[str, Road] = field(default_factory=dict)
    light_states: Dict[str, LightState] = field(default_factory=dict)
    timer: float = 0.0
    phases: List[Tuple[str, ...]] = field(default_factory=list)
    current_phase_index: int = 0
    phase_duration: float = 30.0 # Will be ignored in MPC/Actuated modes
    yellow_time: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    all_red_time: Dict[Tuple[str, ...], float] = field(default_factory=dict)

    def build_phases(self):
        dirs = set(self.incoming_roads.keys())
        phases = []
        # prefer NS and EW pairs when available
        ns = tuple(d for d in ('north', 'south') if d in dirs)
        if ns:
            phases.append(ns)
        ew = tuple(d for d in ('east', 'west') if d in dirs)
        if ew:
            phases.append(ew)
        # if there are remaining single directions, put them as individual phases
        remaining = tuple(sorted(dirs - set(sum((list(p) for p in phases), []))))
        for d in remaining:
            phases.append((d,))
        self.phases = phases
        self.current_phase_index = 0

    def initialize_light_states(self):
        for d in list(self.incoming_roads.keys()):
            self.light_states[d] = LightState.RED
        if self.phases:
            cur = self.phases[self.current_phase_index]
            for d in cur:
                if d in self.light_states:
                    self.light_states[d] = LightState.GREEN
            self.timer = 0.0

    def get_current_phase(self) -> Tuple[str, ...]:
        return self.phases[self.current_phase_index] if self.phases else tuple()

    def get_next_phase_index(self) -> int:
        return (self.current_phase_index + 1) % len(self.phases) if self.phases else 0
    
    def get_next_phase_dirs(self) -> Tuple[str, ...]:
        idx = self.get_next_phase_index()
        return self.phases[idx] if self.phases else tuple()


    def _calculate_yellow_all_red_times(self, reaction_time: float, deceleration: float, vehicle_length: float):
        w = 20.0  # Intersection width

        for i, current_phase in enumerate(self.phases):
            max_yellow, max_all_red = 0.0, 0.0
            next_phase_index = (i + 1) % len(self.phases)
            next_phase_dirs = self.phases[next_phase_index]

            # Yellow time calculation (using kinematic equations for clearance distance)
            for d in current_phase:
                if road := self.incoming_roads.get(d):
                    # v = design speed or approach speed
                    v = max(1.0, road.approach_speed)
                    # t_yellow = perception-reaction time + stopping distance / (2*deceleration)
                    yellow = reaction_time + v / (2.0 * (deceleration + max(0.1, abs(road.grade))))
                    max_yellow = max(max_yellow, yellow)

            # All-red time calculation (clearing the intersection)
            for d in next_phase_dirs:
                if road := self.incoming_roads.get(d):
                    v = max(1.0, road.approach_speed)
                    # All-red time = (Width + Vehicle Length) / Speed
                    all_red = (w + vehicle_length) / v
                    max_all_red = max(max_all_red, all_red)

            self.yellow_time[current_phase] = max(3.0, max_yellow)
            self.all_red_time[current_phase] = max(1.0, max_all_red)


class TrafficPredictor:
    """Weather-aware Traffic Predictor using a Random Forest."""

    def __init__(self, history_steps: int = 5, n_estimators: int = 250, random_state: int = 42):
        self.history_steps = int(history_steps)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False

    def prepare_data(self, historical_df: pd.DataFrame, prediction_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        df = historical_df.copy()
        
        # --- Data preparation (simplified for this context) ---
        if 'downstream_value' not in df.columns:
            df['downstream_value'] = df['value'].rolling(window=3, min_periods=1).mean().shift(-1).fillna(method='ffill').fillna(0.0)
        if 'hour' not in df.columns:
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            else:
                df['hour'] = 0
        if 'rain' not in df.columns:
            df['rain'] = 0
        if 'temperature' not in df.columns:
            df['temperature'] = 20.0

        hours = df['hour'].fillna(0).astype(int).values.flatten()
        rain = df['rain'].fillna(0).values.flatten()
        hist_values = df['value'].ffill().bfill().fillna(0).values.flatten()
        downstream_values = df['downstream_value'].ffill().bfill().fillna(0).values.flatten()
        temperature = df['temperature'].ffill().bfill().fillna(20.0).values.flatten()
        
        # --- Sequence generation ---
        max_i = len(hist_values) - self.history_steps - prediction_steps + 1
        X, Y = [], []

        for i in range(max(0, max_i)):
            X_hist = hist_values[i: i + self.history_steps].tolist()
            X_down = downstream_values[i: i + self.history_steps].tolist()

            last_idx = i + self.history_steps - 1
            X_hour = [int(hours[last_idx])]
            X_rain = [float(rain[last_idx])]
            X_temp = [float(temperature[last_idx])]

            feature_row = X_hist + X_down + X_hour + X_rain + X_temp
            target = hist_values[i + self.history_steps: i + self.history_steps + prediction_steps]

            X.append(feature_row)
            Y.append(target)

        if len(X) == 0:
            return np.empty((0, 2 * self.history_steps + 3)), np.empty((0, prediction_steps))

        return np.array(X), np.array(Y)

    def train(self, historical_df: pd.DataFrame):
        X, Y = self.prepare_data(historical_df, prediction_steps=1)

        if len(X) > 10:
            X = X.reshape(X.shape[0], -1)
            Y = Y.reshape(-1)

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            Y_train, Y_test = Y[:split_idx], Y[split_idx:]

            self.model.fit(X_train, Y_train)
            self.is_trained = True

            try:
                train_score = self.model.score(X_train, Y_train)
                test_score = self.model.score(X_test, Y_test)
                logging.info(f"TrafficPredictor trained. train R^2={train_score:.3f}, test R^2={test_score:.3f}")
            except Exception:
                pass
        else:
            logging.warning("Not enough data to train predictor. Need >10 samples.")

    def predict_congestion(self, current_history: np.ndarray, downstream_history: np.ndarray,
                             current_hour: int, rain: float, temperature: float) -> float:
        """Predicts the next congestion value (queue length or delay)."""
        if not self.is_trained:
            return float(np.mean(current_history)) if len(current_history) > 0 else 50.0

        current_history = np.asarray(current_history).flatten()
        downstream_history = np.asarray(downstream_history).flatten()

        # Simple check for feature vector length
        if len(current_history) != self.history_steps or len(downstream_history) != self.history_steps:
            return float(np.mean(current_history)) if len(current_history) > 0 else 50.0

        feature_vec = np.concatenate([
            current_history, downstream_history, [int(current_hour)], [float(rain)], [float(temperature)]
        ])

        pred = float(self.model.predict(feature_vec.reshape(1, -1))[0])
        return max(0.0, pred)

class SyntheticDataGenerator:
    """Generates synthetic time series data for traffic prediction."""
    
    @staticmethod
    def generate_multi_seasonal(n_days: int, base_value: float, daily_amplitude: float, weekly_amplitude: float, noise_level: float, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        n_steps = n_days * 24 # Hourly data
        timestamps = pd.to_datetime(pd.date_range(start='2024-01-01', periods=n_steps, freq='h'))
        
        daily_cycle = daily_amplitude * np.sin(2 * np.pi * (timestamps.hour - 8) / 24)
        
        # Weekly cycle: high on weekdays (Mon-Fri), low on weekends (Sat-Sun)
        day_of_week = timestamps.dayofweek
        weekly_factor = np.where(day_of_week < 5, 1, 0.6)
        weekly_cycle = weekly_amplitude * (weekly_factor - np.mean(weekly_factor))
        
        noise = np.random.normal(0, noise_level, n_steps)
        
        value = base_value + daily_cycle + weekly_cycle + noise
        value = np.maximum(0, value)

        df = pd.DataFrame({'timestamp': timestamps, 'value': value})
        df['hour'] = df['timestamp'].dt.hour
        return df

    @staticmethod
    def add_weather_effects(df: pd.DataFrame, weather_impact: float = 0.15, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        df['rain'] = 0
        df['temperature'] = 20.0 # Default temp
        
        # Simulate rain during peak hours/seasons
        rainy_indices = df.sample(frac=0.1, random_state=seed).index
        df.loc[rainy_indices, 'rain'] = np.random.choice([1, 2], size=len(rainy_indices))
        df['value'] -= df['rain'] * df['value'] * 0.1 # Traffic decreases with rain

        # Simulate temperature fluctuations (e.g., lower traffic in extreme temps)
        df['temperature'] = 15 + 10 * np.sin(2 * np.pi * df.index / (365*24)) + np.random.normal(0, 5, len(df))
        temp_impact = np.where(df['temperature'] < 5, 0.1, 0) + np.where(df['temperature'] > 30, 0.05, 0)
        df['value'] -= df['value'] * temp_impact
        
        df['value'] = np.maximum(0, df['value'])
        return df


class FixedTrafficController:
    """Baseline controller using fixed green times."""
    
    def get_action(self, intersection: Intersection, sim_time: float):
        current_phase = intersection.get_current_phase()
        # Default duration for fixed time is 30.0 seconds
        if intersection.timer >= 30.0:
            return 'switch_to_yellow'
        return 'hold'

class ActuatedMLTrafficController:
    """A basic actuated controller using the predictor to extend green time."""
    
    def __init__(self, predictor, simulator):
        self.predictor = predictor
        self.simulator = simulator
        self.max_green = 60.0 # Upper limit for green time
        self.min_green = 15.0 # Lower limit

    def get_action(self, intersection: Intersection, sim_time: float):
        current_phase = intersection.get_current_phase()
        current_duration = intersection.timer
        
        # Check if minimum green time has passed
        if current_duration < self.min_green:
            return 'hold'

        # Check for maximum green time
        if current_duration >= self.max_green:
            return 'switch_to_yellow'

        # Predictive actuation logic: check if next phase has high predicted demand
        next_phase_dirs = intersection.get_next_phase_dirs()
        
        # Get predicted queue for the next phase
        max_next_pred = 0.0
        for direction in next_phase_dirs:
            road = intersection.incoming_roads.get(direction)
            if road:
                # Use current queue length as a simple proxy for congestion prediction
                max_next_pred = max(max_next_pred, road.queue_len)

        # If next phase has significant predicted queue, terminate current green
        # Simple threshold: if next phase queue is > 3 vehicles (15.0 units) and we are past min green
        if max_next_pred > 15.0 and current_duration >= self.min_green:
            return 'switch_to_yellow'

        return 'hold'


def _get_road_history(controller, road: Optional['Road']) -> list[float]:
    """Return the last H queue lengths for a road."""
    H = controller.H
    if road and hasattr(road, "history_queue") and len(road.history_queue) >= H:
        return list(road.history_queue[-H:])
    elif road:
        # Fallback: replicate current queue length
        return [float(getattr(road, "queue_len", 0.0))] * H
    else:
        return [0.0] * H

def _get_downstream_road(controller, road: 'Road') -> Optional['Road']:
    """Find the downstream road with largest queue at the next intersection."""
    end_pos = getattr(road, "end_pos", None)
    if end_pos is None: return None
    nxt_inter = next((it for it in controller.simulator.intersections if it.position == end_pos), None)
    if nxt_inter is None: return None
    
    candidates = [r for r in nxt_inter.incoming_roads.values() if r is not road]
    if not candidates: return None
    
    return max(candidates, key=lambda r: getattr(r, "queue_len", 0.0))



class MPCTrafficController:
    """
    UPDATED: Model Predictive Control (MPC) Controller using Sequential Lookahead.
    It evaluates multiple multi-step phase plans (e.g., P1 -> P2 -> P3) 
    to find the sequence with the minimum predicted cost over the horizon.
    """
    
    def __init__(self, predictor: 'TrafficPredictor', simulator: Any, 
                 lookahead_phases: int = 3, phase_duration_guess: float = 15.0):
        self.predictor = predictor
        self.simulator = simulator
        self.H = int(getattr(self.predictor, "history_steps", 5))
        # New parameters for optimization
        self.lookahead_phases = lookahead_phases         # How many future phase changes to consider
        self.phase_duration_guess = phase_duration_guess # Assumed time each phase will run for optimization
        self.MAX_GREEN = 60.0
        self.MIN_GREEN = 15.0

    def _predict_cost(self, intersection: 'Intersection', sim_time: float, 
                      phase_dirs: Tuple[str, ...], duration: float, 
                      predicted_queue_state: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Predict the cost (total queue) and the resulting queue state 
        if the specified phase runs for 'duration'.
        
        :param predicted_queue_state: The queue lengths from the previous prediction step.
        :returns: (Total predicted cost, New predicted queue state)
        """
        predicted_cost = 0.0
        new_queue_state = predicted_queue_state.copy()
        
        sim_hour = int(sim_time / 3600) % 24
        rain = float(self.simulator.current_weather.get('rain', 0))
        temperature = float(self.simulator.current_weather.get('temperature', 20.0))

        # We assume the predictor predicts the *demand* based on current context.
        # This demand interacts with the 'duration' and 'phase_dirs' (control input).
        
        for direction, road in intersection.incoming_roads.items():
            
            # 1. Get current history for ML predictor (using true history for the *start* of the plan)
            hist = _get_road_history(self, road)
            downstream_road = _get_downstream_road(self, road)
            downstream_hist = _get_road_history(self, downstream_road)

            # 2. Predict the queue *if no control action was taken* (i.e., open-loop demand)
            pred_demand = self.predictor.predict_congestion(
                current_history=np.array(hist),
                downstream_history=np.array(downstream_hist),
                current_hour=sim_hour, rain=rain, temperature=temperature
            )

            current_queue = new_queue_state.get(direction, pred_demand)
            
            # 3. Apply control action (GREEN vs. RED) to the predicted queue
            if direction in phase_dirs: # Green Phase
                # Clear queue: Rate is proportional to flow, limited by current queue
                # Assumed clearance rate: 0.5 units/second
                clearance_rate = min(current_queue / duration, 0.5) 
                
                cleared_queue_len = clearance_rate * duration
                resultant_queue = max(0.0, current_queue - cleared_queue_len)
                
                # Cost is incurred by the residual queue after clearance
                cost = resultant_queue * duration # Total queue-time cost
                
            else: # Red/Yellow/All-Red Phase
                # Queue accumulates: Rate is proportional to demand and duration
                # Assumed arrival rate: 0.4 units/second (slightly less than clearance)
                arrival_rate = pred_demand * 0.005 # Scale demand prediction
                
                accumulated_queue_len = arrival_rate * duration 
                resultant_queue = current_queue + accumulated_queue_len
                
                # Cost is incurred by the entire accumulated queue
                cost = resultant_queue * duration
            
            predicted_cost += cost
            new_queue_state[direction] = resultant_queue
            
        return predicted_cost, new_queue_state

    def get_action(self, intersection: 'Intersection', sim_time: float):
        
        current_phase = intersection.get_current_phase()
        current_timer = intersection.timer
        
        if not current_phase or not self.predictor.is_trained:
            return 'hold'
        
        # 1. Safety Checks (Prioritized over Optimization)
        current_state = intersection.light_states.get(current_phase[0], LightState.ALL_RED)
        if current_state != LightState.GREEN:
            # Let the simulator complete the Yellow/All-Red transition
            return 'hold'
        if current_timer < self.MIN_GREEN:
            # Must satisfy minimum green time
            return 'hold'

        # --- 2. MPC Lookahead Optimization ---
        
        best_cost = float('inf')
        optimal_first_action = 'hold'

        # Get the phases list for cyclic planning
        phases_list = intersection.phases
        current_phase_index = intersection.current_phase_index
        
        # Start state for optimization is the actual current queue length
        initial_queue_state = {d: intersection.incoming_roads[d].queue_len 
                               for d in intersection.incoming_roads.keys()}

        # Option A: Policy 1 = HOLD (continue current phase)
        # Option B: Policy 1 = SWITCH (move to next phase)
        
        # Iterate over possible first moves (HOLD or SWITCH)
        for first_move in ['HOLD', 'SWITCH']:
            
            # Initialize for this policy path
            total_plan_cost = 0.0
            simulated_time = sim_time
            simulated_queue_state = initial_queue_state.copy()
            simulated_phase_idx = current_phase_index

            # --- Step 1: Evaluate the first move ---
            
            if first_move == 'HOLD':
                # Policy 1: Current phase continues for self.phase_duration_guess
                p1_dirs = phases_list[simulated_phase_idx]
                p1_duration = self.phase_duration_guess 
            else: # SWITCH
                # Policy 1: Current phase ends now, transition (Y/AR) occurs, then next phase starts.
                # We model the transition as a penalty/cost and assume the next phase is immediately green.
                
                # Cost of transition (Y/AR)
                y_ar_time = intersection.yellow_time.get(current_phase, 3.0) + intersection.all_red_time.get(current_phase, 1.0)
                
                # Calculate the cost of all roads being RED during the transition
                transition_cost, simulated_queue_state = self._predict_cost(
                    intersection, simulated_time, tuple(), y_ar_time, simulated_queue_state
                )
                total_plan_cost += transition_cost
                simulated_time += y_ar_time
                
                # Now model the *next* phase running (this is the first control phase of the SWITCH path)
                simulated_phase_idx = intersection.get_next_phase_index()
                p1_dirs = phases_list[simulated_phase_idx]
                p1_duration = self.phase_duration_guess
            
            # Cost of Policy 1 Green Phase
            p1_cost, simulated_queue_state = self._predict_cost(
                intersection, simulated_time, p1_dirs, p1_duration, simulated_queue_state
            )
            total_plan_cost += p1_cost
            simulated_time += p1_duration
            
            
            # --- Steps 2 to N: Evaluate the rest of the lookahead phases ---
            
            # Continue cycling through phases for the lookahead horizon
            for step in range(1, self.lookahead_phases):
                # Always model a transition (Y/AR) before the next phase change
                y_ar_time = intersection.yellow_time.get(p1_dirs, 3.0) + intersection.all_red_time.get(p1_dirs, 1.0)

                # 1. Cost of Transition (All Red)
                transition_cost, simulated_queue_state = self._predict_cost(
                    intersection, simulated_time, tuple(), y_ar_time, simulated_queue_state
                )
                total_plan_cost += transition_cost
                simulated_time += y_ar_time

                # 2. Cost of Next Green Phase
                simulated_phase_idx = (simulated_phase_idx + 1) % len(phases_list)
                current_sim_dirs = phases_list[simulated_phase_idx]
                
                step_cost, simulated_queue_state = self._predict_cost(
                    intersection, simulated_time, current_sim_dirs, self.phase_duration_guess, simulated_queue_state
                )
                total_plan_cost += step_cost
                simulated_time += self.phase_duration_guess
                
            # --- 3. Compare and Select Optimal First Action ---
            
            if total_plan_cost < best_cost:
                best_cost = total_plan_cost
                
                # The optimal first action is the one that starts this path
                if first_move == 'SWITCH':
                    optimal_first_action = 'switch_to_yellow'
                else:
                    optimal_first_action = 'hold'

        # If the best move is HOLD, check against Max Green time limit
        if optimal_first_action == 'hold' and current_timer >= self.MAX_GREEN:
             return 'switch_to_yellow'
             
        # Execute the optimal first action of the best sequence
        return optimal_first_action



class TrafficSimulator:
    next_vehicle_id = 0
    LANE_OFFSET = 4.0

    DRIVER_REACTION_TIME = 1.0
    DECELERATION_RATE = 3.0
    VEHICLE_LENGTH = 5.0

    def __init__(self, num_intersections: int = 4, road_len: float = 100.0,
                 controller_type: str = 'mpc', use_weather_effects: bool = True):
        self.num_intersections = num_intersections
        self.road_len = road_len
        self.intersections = self._create_intersection_grid(num_intersections)
        self.vehicles: List[Vehicle] = []
        self.time = 0.0
        self.use_weather_effects = use_weather_effects
        self.current_weather = {'rain': 0, 'temperature': 20, 'wind_speed': 10}
        self.collided = False
        self.metrics = {'total_wait_time': 0.0, 'avg_wait_time': 0.0, 'throughput': 0, 'collisions': 0, 'total_simulation_time': 0.0}

        # Generate enhanced historical data with weather effects
        traffic_df = SyntheticDataGenerator.generate_multi_seasonal(
            n_days=30,
            base_value=100.0,
            daily_amplitude=40.0,
            weekly_amplitude=25.0,
            noise_level=15.0,
            seed=42
        )

        if use_weather_effects:
            traffic_df = SyntheticDataGenerator.add_weather_effects(
                traffic_df,
                weather_impact=0.15,
                seed=42
            )

        if 'downstream_value' not in traffic_df.columns:
            traffic_df['downstream_value'] = np.roll(traffic_df['value'].rolling(window=3, min_periods=1).mean() * 0.9 + 10.0, -1)
            traffic_df['downstream_value'] = np.maximum(traffic_df['downstream_value'], 0.0)

        if 'hour' not in traffic_df.columns and 'timestamp' in traffic_df.columns:
            traffic_df['hour'] = pd.to_datetime(traffic_df['timestamp']).dt.hour

        self.predictor = TrafficPredictor(history_steps=5)
        
        try:
            self.predictor.train(traffic_df)
        except Exception as e:
            logging.warning(f"Predictor training failed, continuing without trained model: {e}")

        # Controller selection
        if controller_type == 'fixed':
            self.controller = FixedTrafficController()
        elif controller_type == 'mpc':
            self.controller = MPCTrafficController(self.predictor, simulator=self)
        elif controller_type == 'actuated':
            self.controller = ActuatedMLTrafficController(self.predictor, simulator=self)
        else:
            raise ValueError("controller_type must be 'actuated', 'mpc' or 'fixed'")
        
        logging.info(f"Simulator initialized with controller type: {controller_type}")


    def _create_intersection_grid(self, n: int) -> List[Intersection]:
        intersections: List[Intersection] = []
        grid_dim = int(math.ceil(math.sqrt(n)))

        for i in range(n):
            x, y = i % grid_dim, i // grid_dim
            pos = (x * self.road_len, y * self.road_len)
            intersections.append(Intersection(id=i, position=pos))

        for inter in intersections:
            x_idx = int(inter.position[0] / self.road_len)
            y_idx = int(inter.position[1] / self.road_len)

            neighbors = {
                'north': (x_idx, y_idx + 1),
                'south': (x_idx, y_idx - 1),
                'east':  (x_idx + 1, y_idx),
                'west':  (x_idx - 1, y_idx)
            }

            for direction, (nx, ny) in neighbors.items():
                if 0 <= nx < grid_dim and 0 <= ny < grid_dim:
                    neighbor_idx = nx + ny * grid_dim
                    if neighbor_idx < len(intersections):
                        neighbor = intersections[neighbor_idx]

                        approach_speed = 10.0

                        # Create road FROM neighbor TO current intersection
                        inter.incoming_roads[direction] = Road(
                            name=f"Road_{inter.id}_{direction}",
                            length=self.road_len,
                            start_pos=neighbor.position,
                            end_pos=inter.position,
                            approach_speed=approach_speed,
                            grade=0.0
                        )

            inter.build_phases()
            inter._calculate_yellow_all_red_times(
                reaction_time=self.DRIVER_REACTION_TIME,
                deceleration=self.DECELERATION_RATE,
                vehicle_length=self.VEHICLE_LENGTH
            )
            inter.initialize_light_states()

        return intersections

    def _get_all_roads(self):
        roads = []
        for inter in self.intersections:
            roads.extend(inter.incoming_roads.values())
        return roads

    def update_weather_conditions(self):
        if not self.use_weather_effects:
            return

        sim_hours = self.time / 3600
        hour_of_day = int(sim_hours) % 24
        day_of_year = int(sim_hours / 24) % 365

        rain_prob = 0.1
        if 14 <= hour_of_day <= 18:
            rain_prob = 0.3
        if 60 <= day_of_year <= 150 or 240 <= day_of_year <= 330:
            rain_prob *= 1.5

        if random.random() < rain_prob:
            self.current_weather['rain'] = random.choice([1, 2])
        else:
            self.current_weather['rain'] = 0

        base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        daily_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
        self.current_weather['temperature'] = max(-10, base_temp + daily_variation + random.uniform(-3, 3))

        # apply to roads
        for road in self._get_all_roads():
            self._update_road_for_weather(road)

    def _update_road_for_weather(self, road: Road):
        # adjust approach_speed based on rain
        rain_intensity = int(self.current_weather.get('rain', 0))
        if rain_intensity == 1:
            road.approach_speed = max(3.0, 0.8 * 10.0)
        elif rain_intensity == 2:
            road.approach_speed = max(3.0, 0.6 * 10.0)
        else:
            road.approach_speed = 10.0

        # temperature effects (icy)
        temp = float(self.current_weather.get('temperature', 20.0))
        if temp < 0:
            road.approach_speed = min(road.approach_speed, 5.0)
            road.grade = 0.1
        else:
            # reset grade to default when not icy
            road.grade = 0.0

    def get_current_traffic_pattern(self) -> float:
        sim_hours = self.time / 3600
        hour_of_day = int(sim_hours) % 24
        day_of_week = int(sim_hours / 86400) % 7

        base_traffic = 100.0
        if 7 <= hour_of_day <= 9:
            morning_peak = 40.0 * (1 - abs(hour_of_day - 8) / 2)
            base_traffic += morning_peak
        elif 16 <= hour_of_day <= 19:
            evening_peak = 50.0 * (1 - abs(hour_of_day - 17.5) / 2.5)
            base_traffic += evening_peak

        if day_of_week >= 5:
            base_traffic *= 0.7

        if self.use_weather_effects:
            rain_effect = float(self.current_weather.get('rain', 0)) * 10.0
            temp_effect = -abs(float(self.current_weather.get('temperature', 20.0)) - 20) * 0.5
            base_traffic += rain_effect + temp_effect

        return max(20.0, base_traffic)

    def spawn_vehicle(self, dt: float = 0.1):
        base_spawn_rate = 1.5
        traffic_factor = self.get_current_traffic_pattern() / 100.0

        weather_factor = 1.0
        if self.use_weather_effects:
            if self.current_weather['rain'] > 0:
                weather_factor *= 0.8
            if self.current_weather['temperature'] < 0:
                weather_factor *= 0.7

        adjusted_spawn_rate = base_spawn_rate * traffic_factor * weather_factor
        num_spawns = np.random.poisson(max(0.0, adjusted_spawn_rate * dt))

        for _ in range(num_spawns):
            candidates = [i for i in self.intersections if len(i.incoming_roads) > 0]
            if not candidates:
                return

            start_int = random.choice(candidates)
            start_dir = random.choice(list(start_int.incoming_roads.keys()))
            start_road = start_int.incoming_roads[start_dir]

            possible_dests = [i for i in self.intersections if i.id != start_int.id]
            if not possible_dests:
                return
            end_int = random.choice(possible_dests)

            lane_offset_choice = self.LANE_OFFSET
            sx, sy = start_road.start_pos
            ex, ey = start_road.end_pos
            rv = np.array([ex - sx, ey - sy], dtype=float)
            norm = np.linalg.norm(rv)
            if norm < 1e-6:
                continue

            u = rv / norm
            perp = np.array([-u[1], u[0]])
            pos = (sx + perp[0] * lane_offset_choice, sy + perp[1] * lane_offset_choice)

            # Check for immediate occupancy
            occupied = any(np.linalg.norm(np.array(v.position) - np.array(pos)) < 6.0 for v in self.vehicles)
            if occupied:
                continue

            base_speed = float(np.random.uniform(5.0, 10.0))
            if self.use_weather_effects:
                if self.current_weather['rain'] > 0:
                    base_speed *= 0.8
                if self.current_weather['temperature'] < 0:
                    base_speed *= 0.7

            v = Vehicle(
                id=TrafficSimulator.next_vehicle_id,
                position=pos,
                destination=end_int.position,
                speed=base_speed,
                max_speed=base_speed * 1.2,
                current_road=start_road,
                next_intersection=start_int,
                lane_offset=lane_offset_choice
            )
            self.vehicles.append(v)
            TrafficSimulator.next_vehicle_id += 1

    def _route_vehicle(self, vehicle: Vehicle) -> bool:
        current_int = vehicle.next_intersection
        if current_int is None or vehicle.current_road is None:
            return False
        incoming_dir = next((d for d, r in current_int.incoming_roads.items() if r is vehicle.current_road), None)
        if not incoming_dir:
            return False

        # Simple routing logic: choose a random turn
        TURNS = {
            'north': {'straight': 'south', 'right': 'west', 'left': 'east'},
            'south': {'straight': 'north', 'right': 'east', 'left': 'west'},
            'east':  {'straight': 'west',  'right': 'north', 'left': 'south'},
            'west':  {'straight': 'east',  'right': 'south', 'left': 'north'},
        }
        OPPOSITE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        
        selected_turn_name = random.choices(['straight', 'right', 'left'], weights=[0.7, 0.15, 0.15], k=1)[0]
        outgoing_exit_dir = TURNS[incoming_dir][selected_turn_name]
        road_len = self.road_len # All roads are the same length in this grid
        x_int, y_int = current_int.position

        # Calculate target position for the next intersection
        if outgoing_exit_dir == 'north':    target_pos = (x_int, y_int + road_len)
        elif outgoing_exit_dir == 'south':  target_pos = (x_int, y_int - road_len)
        elif outgoing_exit_dir == 'east':   target_pos = (x_int + road_len, y_int)
        elif outgoing_exit_dir == 'west':   target_pos = (x_int - road_len, y_int)
        else: return False

        target_int = next((i for i in self.intersections if i.position == target_pos), None)

        if target_int:
            new_road_dir = OPPOSITE[outgoing_exit_dir]
            new_road = target_int.incoming_roads.get(new_road_dir)

            if new_road:
                # Place vehicle at the start of the new road
                sx, sy = new_road.start_pos
                ex, ey = new_road.end_pos
                rv = np.array([ex - sx, ey - sy], dtype=float)
                rn = np.linalg.norm(rv)

                if rn > 1e-6:
                    u = rv / rn
                    perp = np.array([-u[1], u[0]])
                    pos = np.array(new_road.start_pos) + perp * vehicle.lane_offset
                    pos = (float(pos[0]), float(pos[1]))
                else:
                    pos = new_road.start_pos

                vehicle.current_road = new_road
                vehicle.next_intersection = target_int
                vehicle.position = pos
                vehicle.speed = float(np.random.uniform(5.0, 10.0))
                vehicle.wait_time = 0.0
                return True

        return False

    def update_lights(self, dt: float):
        for inter in self.intersections:
            current_phase = inter.get_current_phase()
            
            # --- Controller Action ---
            # Controller can only request a switch from GREEN to YELLOW
            action = self.controller.get_action(inter, self.time)

            # increment timer first so durations are measured from when state was entered
            inter.timer += dt

            yellow_duration = inter.yellow_time.get(current_phase, 3.0)
            all_red_duration = inter.all_red_time.get(current_phase, 1.0)

            # If current phase signals are GREEN
            if current_phase and all(inter.light_states.get(d) == LightState.GREEN for d in current_phase):
                # The controller requested a switch (MPC decision)
                if action == 'switch_to_yellow':
                    for d in current_phase:
                        inter.light_states[d] = LightState.YELLOW
                    # The switch transition starts now
                    inter.timer = 0.0

            # If current phase signals are YELLOW
            elif current_phase and all(inter.light_states.get(d) == LightState.YELLOW for d in current_phase):
                if inter.timer >= yellow_duration:
                    # Time to transition to ALL RED
                    for d in current_phase:
                        inter.light_states[d] = LightState.ALL_RED
                    inter.timer = 0.0

            # If current phase signals are ALL_RED
            elif current_phase and all(inter.light_states.get(d) == LightState.ALL_RED for d in current_phase):
                if inter.timer >= all_red_duration:
                    # Time to advance phase and apply greens
                    inter.current_phase_index = inter.get_next_phase_index()
                    next_dirs = inter.get_current_phase()
                    
                    # Set the new phase to GREEN, others to RED
                    for d in list(inter.light_states.keys()):
                        inter.light_states[d] = LightState.GREEN if d in next_dirs else LightState.RED
                    inter.timer = 0.0
                    
            # If the intersection just started or is in an idle state, ensure the first phase is set
            else:
                 inter.initialize_light_states()


    def step(self, dt: float = 0.1):
        
        # 1. Update weather
        self.update_weather_conditions()

        # 2. Update lights (controller decisions)
        self.update_lights(dt)

        # 3. Spawn new vehicles
        self.spawn_vehicle(dt)

        vehicles_to_remove: List[Vehicle] = []
        
        # Parameters for safe-following
        SAFE_GAP = 5.0
        RANDOM_COLLISION_PROB = 0.0001 

        # 4. Update vehicle positions and road queues
        for road in self._get_all_roads():
            # Update history for predictor
            road.history_queue.append(road.queue_len)
            if len(road.history_queue) > 100: # Limit history size
                road.history_queue.pop(0)

            # Re-sort queue based on distance to stop line (end_pos)
            road.queue = [v for v in self.vehicles if v.current_road is road]
            road.queue.sort(key=lambda v: np.linalg.norm(np.array(v.position) - np.array(road.end_pos)))
            road.queue_len = len(road.queue) * self.VEHICLE_LENGTH # Use vehicle length as a proxy for queue length

        
        # 5. Move Vehicles
        for vehicle in list(self.vehicles):
            road = vehicle.current_road
            next_int = vehicle.next_intersection

            if road is None or next_int is None:
                vehicles_to_remove.append(vehicle)
                continue

            sx, sy = road.start_pos
            ex, ey = road.end_pos
            road_vec = np.array([ex - sx, ey - sy], dtype=float)
            road_len = np.linalg.norm(road_vec)
            
            if road_len < 1e-6:
                vehicles_to_remove.append(vehicle)
                continue

            unit = road_vec / road_len
            rel = np.array(vehicle.position) - np.array(road.start_pos)
            longitudinal = float(np.dot(rel, unit))
            dist_to_stop_line = (road.stop_line() - longitudinal)

            road_dir = next((d for d, r in next_int.incoming_roads.items() if r is road), None)
            light_state = next_int.light_states.get(road_dir, LightState.RED) if road_dir else LightState.RED

            target_speed = vehicle.max_speed
            is_stopped = False

            # Calculate safe speed based on light/stop line
            if dist_to_stop_line < 5.0 and light_state != LightState.GREEN:
                # Must stop for red/yellow/all-red
                target_speed = 0.0
                
            elif dist_to_stop_line < 30.0 and light_state != LightState.GREEN:
                # Decelerate when approaching stop line for non-green light
                target_speed = min(target_speed, vehicle.max_speed * (dist_to_stop_line / 30.0))

            # IDM-based safe-following (simple car-following model)
            try:
                current_index = road.queue.index(vehicle)
                if current_index > 0:
                    leader = road.queue[current_index - 1]
                    rel_gap = np.linalg.norm(np.array(leader.position) - np.array(vehicle.position)) - self.VEHICLE_LENGTH
                    
                    if rel_gap < SAFE_GAP + 0.5 * vehicle.speed: # Time headway safety gap
                        # Adjust target speed to match leader or stop
                        target_speed = min(target_speed, max(0.0, leader.speed - 0.5))
                        
                    if rel_gap < 1.0 and random.random() < RANDOM_COLLISION_PROB:
                        self.metrics['collisions'] += 1
                        self.collided = True

            except ValueError:
                pass # Vehicle not found in queue (shouldn't happen often)

            # Apply acceleration/deceleration to reach target_speed
            if target_speed > vehicle.speed:
                acceleration = 2.0 # Max comfortable acceleration
            else:
                acceleration = -self.DECELERATION_RATE # Max safe deceleration

            vehicle.speed = max(0.0, vehicle.speed + acceleration * dt)
            vehicle.speed = min(vehicle.max_speed, vehicle.speed)
            
            # Update wait time if speed is near zero and not at destination
            if vehicle.speed < 0.1 and dist_to_stop_line > 0:
                 vehicle.wait_time += dt

            # Update position
            movement = unit * vehicle.speed * dt
            longitudinal_movement = float(np.dot(movement, unit))
            new_longitudinal = longitudinal + longitudinal_movement
            
            perp = np.array([-unit[1], unit[0]])
            base_point = np.array(road.start_pos) + unit * new_longitudinal
            new_pos = base_point + perp * vehicle.lane_offset
            vehicle.position = (float(new_pos[0]), float(new_pos[1]))
            
            long_after = float(np.dot(np.array(vehicle.position) - np.array(road.start_pos), unit))


            # Check if vehicle reached end of road (entered intersection area)
            if long_after >= road.length:
                if next_int.position == vehicle.destination:
                    # Reached final destination
                    self.metrics['throughput'] += 1
                    self.metrics['total_wait_time'] += vehicle.wait_time
                    vehicles_to_remove.append(vehicle)
                else:
                    # Route to the next road
                    routed = self._route_vehicle(vehicle)
                    if not routed:
                        # Dead end or routing failure, remove vehicle
                        self.metrics['throughput'] += 1
                        self.metrics['total_wait_time'] += vehicle.wait_time
                        vehicles_to_remove.append(vehicle)

        # 6. Final cleanup
        self.vehicles = [v for v in self.vehicles if v not in vehicles_to_remove]
        self.time += dt
        self.metrics['total_simulation_time'] = self.time

        if self.metrics['throughput'] > 0:
            self.metrics['avg_wait_time'] = self.metrics['total_wait_time'] / self.metrics['throughput']


    def get_metrics(self):
        return self.metrics


class TrafficVisualizer:
    def __init__(self, simulator: TrafficSimulator, dt: float = 0.5):
        self.simulator = simulator
        self.dt = dt
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.vehicles_patches = {}
        self.lights_patches = {}
        self.road_len = simulator.road_len

        self._setup_plot()

    def _setup_plot(self):
        self.ax.set_xlim(-self.road_len, self.road_len * 2)
        self.ax.set_ylim(-self.road_len, self.road_len * 2)
        self.ax.set_aspect('equal')
        self.ax.set_title("Traffic Simulation")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")

        # Draw roads
        for inter in self.simulator.intersections:
            for road in inter.incoming_roads.values():
                sx, sy = road.start_pos
                ex, ey = road.end_pos
                self.ax.plot([sx, ex], [sy, ey], color='gray', linewidth=2, zorder=0)

        # Draw intersections as squares
        for inter in self.simulator.intersections:
            x, y = inter.position
            rect = patches.Rectangle((x - 3, y - 3), 6, 6, facecolor='lightgray', edgecolor='black', zorder=1)
            self.ax.add_patch(rect)

        # Initialize traffic lights patches
        for inter in self.simulator.intersections:
            self.lights_patches[inter.id] = {}
            for direction in inter.incoming_roads.keys():
                x, y = inter.position
                dx, dy = 0, 0
                if direction == 'north': dy = 4
                elif direction == 'south': dy = -4
                elif direction == 'east': dx = 4
                elif direction == 'west': dx = -4
                light = patches.Circle((x + dx, y + dy), radius=1.0, facecolor='red', edgecolor='black', zorder=2)
                self.ax.add_patch(light)
                self.lights_patches[inter.id][direction] = light

    def _update_vehicles(self):
        # Remove patches for vehicles that no longer exist
        current_ids = set(v.id for v in self.simulator.vehicles)
        for vid in list(self.vehicles_patches.keys()):
            if vid not in current_ids:
                self.vehicles_patches[vid].remove()
                del self.vehicles_patches[vid]

        # Add or update vehicle patches
        for vehicle in self.simulator.vehicles:
            x, y = vehicle.position
            if vehicle.id not in self.vehicles_patches:
                rect = patches.Rectangle((x-1.5, y-1), 3, 2, facecolor='blue', edgecolor='black', zorder=3)
                self.ax.add_patch(rect)
                self.vehicles_patches[vehicle.id] = rect
            else:
                rect = self.vehicles_patches[vehicle.id]
                rect.set_xy((x-1.5, y-1))

    def _update_lights(self):
        for inter in self.simulator.intersections:
            for direction, light_patch in self.lights_patches[inter.id].items():
                state = inter.light_states.get(direction, LightState.RED)
                if state == LightState.GREEN:
                    light_patch.set_facecolor('green')
                elif state == LightState.YELLOW:
                    light_patch.set_facecolor('yellow')
                else:
                    light_patch.set_facecolor('red')

    def _update(self, frame):
        # Step simulator
        self.simulator.step(self.dt)

        # Update vehicle positions
        self._update_vehicles()

        # Update traffic lights
        self._update_lights()

        return list(self.vehicles_patches.values()) + [lp for sub in self.lights_patches.values() for lp in sub.values()]

    def animate(self, duration: float = 60):
        frames = int(duration / self.dt)
        anim = FuncAnimation(self.fig, self._update, frames=frames, interval=self.dt*1000, blit=True)
        plt.show()

# Usage example:
if __name__ == "__main__":

    # 1. Setup Simulation
    SIMULATION_DURATION_SECONDS = 100 
    TIME_STEP = 1

    sim = TrafficSimulator(
        num_intersections=4, 
        road_len=100.0, 
        controller_type='mpc', 
        use_weather_effects=True
    )

    logging.info(f"Starting simulation for {SIMULATION_DURATION_SECONDS / 60} minutes with MPC...")

    # 2. Run Simulation
    current_time = 0.0
    while current_time < SIMULATION_DURATION_SECONDS:
        sim.step(TIME_STEP)
        current_time = sim.time
        
        if int(current_time) % 60 == 0 and int(current_time) != int((current_time - TIME_STEP)):
            # Log metrics every minute
            metrics = sim.get_metrics()
            # logging.info(f"Time: {int(current_time)}s. Throughput: {metrics['throughput']}, Avg Wait: {metrics['avg_wait_time']:.2f}s")
            
    # 3. Final Metrics
    final_metrics = sim.get_metrics()
    print("\n" + "="*40)
    print("      FINAL SIMULATION RESULTS      ")
    print("="*40)
    print(f"Controller Used: {'MPC Traffic Controller'}")
    print(f"Total Vehicles Passed (Throughput): {final_metrics['throughput']}")
    print(f"Total Wait Time: {final_metrics['total_wait_time']:.2f} seconds")
    print(f"Average Wait Time per Vehicle: {final_metrics['avg_wait_time']:.2f} seconds")
    print(f"Total Collisions: {final_metrics['collisions']}")
    print("="*40 + "\n")
    visualizer = TrafficVisualizer(sim, dt=0.5)
    visualizer.animate(duration=120)

    sim = TrafficSimulator(
        num_intersections=4, 
        road_len=100.0, 
        controller_type='fixed', 
        use_weather_effects=True
    )

    logging.info(f"Starting simulation for {SIMULATION_DURATION_SECONDS / 60} minutes with Fixed...")

    # 2. Run Simulation
    current_time = 0.0
    while current_time < SIMULATION_DURATION_SECONDS:
        sim.step(TIME_STEP)
        current_time = sim.time
        
        if int(current_time) % 60 == 0 and int(current_time) != int((current_time - TIME_STEP)):
            # Log metrics every minute
            metrics = sim.get_metrics()
            # logging.info(f"Time: {int(current_time)}s. Throughput: {metrics['throughput']}, Avg Wait: {metrics['avg_wait_time']:.2f}s")
            
    # 3. Final Metrics
    final_metrics = sim.get_metrics()
    print("\n" + "="*40)
    print("      FINAL SIMULATION RESULTS      ")
    print("="*40)
    print(f"Controller Used: {'Fixed Traffic Controller'}")
    print(f"Total Vehicles Passed (Throughput): {final_metrics['throughput']}")
    print(f"Total Wait Time: {final_metrics['total_wait_time']:.2f} seconds")
    print(f"Average Wait Time per Vehicle: {final_metrics['avg_wait_time']:.2f} seconds")
    print(f"Total Collisions: {final_metrics['collisions']}")
    print("="*40 + "\n")

    visualizer = TrafficVisualizer(sim, dt=0.5)
    visualizer.animate(duration=120)  # duration in seconds


