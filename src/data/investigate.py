import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/supply_chain_clean.csv")

# Check actual values in suspicious columns
print("="*50)
print("SHIPPING TIMES vs LEAD TIMES")
print("="*50)
print(df[['shipping_times', 'lead_times']].describe())

print("\n" + "="*50)
print("AVAILABILITY vs ORDER QUANTITIES")
print("="*50)
print(df[['availability', 'order_quantities']].describe())

print("\n" + "="*50)
print("MANUFACTURING COSTS vs INVENTORY VALUE")
print("="*50)
print(df[['manufacturing_costs', 'stock_levels', 'price']].describe())

print("\n" + "="*50)
print("SAMPLE 5 ROWS")
print("="*50)
print(df[['shipping_times', 'lead_times', 
          'availability', 'order_quantities',
          'manufacturing_costs', 'stock_levels', 
          'price']].head())

# Check inventory turnover components
print("="*50)
print("INVENTORY TURNOVER CHECK")
print("="*50)
df['total_cogs'] = df['manufacturing_costs'] * df['number_of_products_sold']
df['inventory_value'] = df['stock_levels'] * df['price']
df['inventory_turnover'] = df['total_cogs'] / df['inventory_value'].replace(0, np.nan)

print(df[['manufacturing_costs', 
          'number_of_products_sold',
          'total_cogs',
          'stock_levels',
          'price',
          'inventory_value',
          'inventory_turnover']].describe())

print("\nSample 5 rows:")
print(df[['total_cogs',
          'inventory_value', 
          'inventory_turnover']].head())

## Checking OTIF score
print("="*50)
print("OTIF INVESTIGATION")
print("="*50)

on_time = df['shipping_times'] <= df['lead_times']
in_full = df['number_of_products_sold'] >= df['order_quantities']

print(f"On Time %:  {on_time.mean():.1%}")
print(f"In Full %:  {in_full.mean():.1%}")
print(f"OTIF %:     {(on_time & in_full).mean():.1%}")

print("\nFailing In Full sample:")
print(df[~in_full][['number_of_products_sold', 
                     'order_quantities']].head(10))

print("\nFailing On Time sample:")
print(df[~on_time][['shipping_times', 
                     'lead_times']].head(10))


