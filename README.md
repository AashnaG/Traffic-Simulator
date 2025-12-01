# Traffic-Simulator
## Phase 2 

To run the code make sure synthetic_data.py and traffic_simulator.py are in the same folder. Then using your command prompt go to the submisson folder and run python traffic_simulator.py.

### Problem: Design an itelligent traffic management system that optimizes traffic flow during rush hour.
### Implemented Solution: 
- A traffic simulator is used to model a grid of intersections with roads, vehicles, and traffic lights at intersections.
- Traffic is simulated based on peak hours (time of day), day of the week (week day or weekend) and weather conditions using the synthetic_data provided.
- Data is generated using the synthetic data generator in order to train a machine learning model (Random Frest Regressor) to predict traffic.
- The predictions of the machine learning model are used to optimize the traffic lights at intersections using model-based predictive control.

### Design Decisions:
- The use of syntheic data allowed for ML training without real traffic data but the data may not capture real-world nuances.
- Simplified vehicle movements and visualisations prioritises simulation speed.
- Random Forests is a simple approach to predicting traffic and doesn't capture temporal dependencies, however introducing a lag feature of previous traffic has the effect of the Random Forest "remembering" past data.
- A threshold function is used to determine the congestions queues at each intersection and the traffic lights are changed based on this threshold function.

### Limitations: 
- Random Forests is a simple method however it does not model temporal dependencies well and despite the feature engineering implemented, an LSTM might out perform the Random Forest Regressor.
- The optimization method tends to only increase the throughput by a small percentage, where the average waiting time increases by a larger percentage. More focus should pu placed on improing the throughput metric. Additionally, the there is a time complexity with the optimization alogrithm.
- Using synthetic data may not represent real world data well.
