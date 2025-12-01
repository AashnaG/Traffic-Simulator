import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import random

class SyntheticDataGenerator:
    """Generate synthetic data with realistic patterns"""
    
    @staticmethod
    def generate_time_series(
        n_days: int = 30,
        base_value: float = 100.0,
        trend: float = 0.0,
        seasonal_period: int = 24,
        seasonal_amplitude: float = 30.0,
        noise_level: float = 10.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data with trend, seasonality, and noise
        
        Useful for: traffic patterns, order volumes, player statistics
        
        Args:
            n_days: Number of days to generate
            base_value: Base level of the series
            trend: Linear trend per hour (positive = increasing)
            seasonal_period: Hours per season (24 = daily pattern)
            seasonal_amplitude: Strength of seasonal pattern
            noise_level: Standard deviation of random noise
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns: timestamp, value
        """
        np.random.seed(seed)
        
        hours = n_days * 24
        timestamps = [
            datetime.now() + timedelta(hours=i) 
            for i in range(hours)
        ]
        
        # Trend component
        t = np.arange(hours)
        trend_component = trend * t
        
        # Seasonal component (e.g., daily pattern)
        seasonal_component = seasonal_amplitude * np.sin(
            2 * np.pi * t / seasonal_period
        )
        
        # Noise component
        noise = np.random.normal(0, noise_level, hours)
        
        # Combine
        values = base_value + trend_component + seasonal_component + noise
        values = np.maximum(values, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    @staticmethod
    def generate_multi_seasonal(
        n_days: int = 30,
        base_value: float = 100.0,
        daily_amplitude: float = 30.0,
        weekly_amplitude: float = 20.0,
        noise_level: float = 10.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate time series with both daily and weekly patterns
        
        Useful for: traffic (weekday vs weekend), orders (weekly cycles)
        """
        np.random.seed(seed)
        
        hours = n_days * 24
        timestamps = [
            datetime.now() + timedelta(hours=i) 
            for i in range(hours)
        ]
        
        t = np.arange(hours)
        
        # Daily pattern (peak at hour 17 = 5pm)
        daily = daily_amplitude * np.sin(2 * np.pi * (t % 24 - 6) / 24)
        
        # Weekly pattern (lower on weekends)
        weekly = weekly_amplitude * np.sin(2 * np.pi * t / (24 * 7))
        
        # Noise
        noise = np.random.normal(0, noise_level, hours)
        
        values = base_value + daily + weekly + noise
        values = np.maximum(values, 0)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        # Add day of week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    @staticmethod
    def generate_poisson_events(
        n_days: int = 30,
        base_rate: float = 10.0,
        peak_hours: List[int] = [8, 9, 17, 18],
        peak_multiplier: float = 2.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate event counts following Poisson distribution with time-varying rate
        
        Useful for: order arrivals, vehicle spawning, random events
        
        Args:
            n_days: Number of days
            base_rate: Base events per hour
            peak_hours: Hours with increased rate
            peak_multiplier: How much higher during peaks
            
        Returns:
            DataFrame with hourly event counts
        """
        np.random.seed(seed)
        
        hours = n_days * 24
        timestamps = [
            datetime.now() + timedelta(hours=i) 
            for i in range(hours)
        ]
        
        event_counts = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            rate = base_rate * peak_multiplier if hour in peak_hours else base_rate
            count = np.random.poisson(rate)
            event_counts.append(count)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'event_count': event_counts,
            'hour': [ts.hour for ts in timestamps]
        })                
    @staticmethod
    def add_weather_effects(
        df: pd.DataFrame,
        weather_impact: float = 0.2,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Add synthetic weather data and its effects
        
        Args:
            df: DataFrame with timestamp column
            weather_impact: How much weather affects the target variable
            
        Returns:
            DataFrame with added weather columns and modified values
        """
        np.random.seed(seed)
        
        # Generate weather conditions
        n_rows = len(df)
        
        # Temperature (Celsius)
        df['temperature'] = np.random.normal(15, 8, n_rows)
        
        # Rain (0=no rain, 1=light, 2=heavy)
        df['rain'] = np.random.choice([0, 1, 2], n_rows, p=[0.7, 0.2, 0.1])
        
        # Wind speed (km/h)
        df['wind_speed'] = np.random.exponential(15, n_rows)
        
        # Weather impact on value (if exists)
        if 'value' in df.columns:
            # Rain increases traffic/delays
            rain_effect = df['rain'] * weather_impact * df['value']
            # Extreme temperature reduces activity
            temp_effect = np.abs(df['temperature'] - 20) * 0.01 * df['value']
            
            df['value'] = df['value'] + rain_effect - temp_effect
            df['value'] = df['value'].clip(lower=0)
        
        return df

def demo_synthetic_data():
    """Demonstrate synthetic data generation"""
    print("Synthetic Data Generator Demo")
    print("=" * 50)
    
    gen = SyntheticDataGenerator()
    
    # Traffic pattern
    print("\n1. Traffic Pattern (multi-seasonal)")
    traffic = gen.generate_multi_seasonal(
        n_days=7,
        base_value=50,
        daily_amplitude=30,
        weekly_amplitude=20
    )
    print(traffic.head())
    print(f"Shape: {traffic.shape}")
    
    # Order arrivals
    print("\n2. Order Arrivals (Poisson events)")
    orders = gen.generate_poisson_events(
        n_days=7,
        base_rate=5,
        peak_hours=[9, 10, 11, 14, 15, 16]
    )
    print(orders.head())
    print(f"Total orders: {orders['event_count'].sum()}")
    
 
    # Weather effects
    print("\n4. Adding Weather Effects")
    traffic_with_weather = gen.add_weather_effects(traffic)
    print(traffic_with_weather.head())
    
    print("\n" + "=" * 50)
    print("Use these functions in your solutions to generate test data!")


if __name__ == "__main__":
    demo_synthetic_data()



if __name__ == "__main__":
    # Add external dependency check        
    demo_synthetic_data()
