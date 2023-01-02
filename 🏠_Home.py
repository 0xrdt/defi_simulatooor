import streamlit as st
from shroomdk import ShroomDK
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from scipy import optimize

# set page to white theme
st.set_page_config(layout="wide", page_title="simulatooor", page_icon="🧠", initial_sidebar_state="collapsed")

# st.markdown(df### ['gas_used'].mean())
#st.write(df)
st.header("Strategy Simulator")

st.markdown("""
Yo!

This app simulates defi strategies in which you:
- you deposit a token (e.g. ETH) as collateral
- then you borrow the same token according to the leverage (e.g. borrow 2x the deposited amount if leverage = 2)
- then you convert part of your assets into a secondary token (e.g. wstETH)
- the amount converted is determined by the liquidity pool ratio (e.g. 50% of the deposited token and 50% of the secondary token)
- then you deposit the tokens into a liquidity pool (e.g. wstETH/ETH)
- the tokens will accrue interest over time, determined by the APY (e.g. 4.75%/year)
- and you will also pay interest on the borrowed tokens (e.g. 2%/year)
- when you exit the strategy, the LP tokens will be redeemed for the two tokens (deposit and secondary token)
- the secondary token will be converted back into the deposited token
- you'll pay off your debt and keep the rest
- your profit will be given by the amount left after paying off the debt - the amount you initially deposited

You can also use this app to simulate simpler strategies if you:
- set the leverage to 0 to ignore the lending and borrowing part (e.g. liquidity pool farming with no leverage)
- set the liquidity pool ratio to 10¨% to ignore the conversion to liquidity pool tokens (e.g. leveraged farming a single token)
- set the APY to 0% to simulate only the borrowing costs

The default parameters are a real world example using the Notional wstETH/ETH leveraged vaults strategy.
""")

st.subheader('Inputs:')
col1, col2 = st.columns(2)
deposited_amount = col1.number_input('Deposited Amount', value=125.0, step=0.0001)
leverage = col2.number_input('Leverage (total asset amount = 1 + leverage)', value=2.0, step=0.01)
initial_price = col1.number_input('Initial Price (secondary token / deposited token price)', value=0.985, step=0.0001)
final_price = col2.number_input('Estimated Final Price (secondary token / deposited token price)', value=0.970, step=0.0001)
estimated_apy = col1.number_input('Estimated average APY (%)', value=5.75, step=0.01)
borrowing_interest_rate = col2.number_input('Estimated average borrowing interest rate (%)', value=5.06, step=0.01)
liquidity_pool_composition = col1.number_input('liquidity pool ratio (use the % of the deposited token)', value=50.0, step=0.01)
days_invested = col2.number_input('days invested', value=60, step=1)

st.subheader("Optional Tx Cost Inputs:")
col1, col2 = st.columns(2)
gas_price = col1.number_input('gas price when depositing', value=16.0, step=0.01, min_value=1.0, max_value=1000.0)
eth_price_vs_deposit_token = col2.number_input('ETH price relative to the deposited token', value=1.0, step=0.000001)
gas_used_entry =  col1.number_input('gas cost of entering the strategy', value=1_250_000, step=1)
gas_used_exit =  col2.number_input('gas cost of exiting the strategy', value=1_250_000, step=1)
gas_cost_entry = gas_price * (gas_used_entry) / 1e9
gas_cost_exit = gas_price * (gas_used_exit) / 1e9

st.subheader("Results:")

# apy (yield compounds)
daily_gain = ((1+estimated_apy/100)**(1/365)-1)
total_gain_pct =  ((1+daily_gain)**days_invested-1)*100
# apr (interest doesn't compound)
# daily_borrowing_interest_rate = ((1+borrowing_interest_rate/100)**(1/365)-1)
# total_interest_pct =  ((1+daily_borrowing_interest_rate)**days_invested-1)*100
daily_borrowing_interest_rate = (borrowing_interest_rate/100)/365
total_interest_pct =  (daily_borrowing_interest_rate*days_invested)*100

initial_asset_amount = deposited_amount * (1 + leverage)
initial_liability_amount = (deposited_amount * leverage) * (1 + total_interest_pct / 100)

initial_deposited_token_pool_amount = initial_asset_amount * liquidity_pool_composition / 100
initial_secondary_token_pool_amount = (initial_asset_amount * (100 - liquidity_pool_composition) / 100)/initial_price

final_deposited_token_pool_amount = initial_deposited_token_pool_amount * (1+total_gain_pct/100)
final_secondary_token_pool_amount = initial_secondary_token_pool_amount * (1+total_gain_pct/100)

final_asset_amount = final_deposited_token_pool_amount + (final_secondary_token_pool_amount * final_price)
final_liability_amount = initial_liability_amount

returns = (
	deposited_amount * 
	(leverage  + 1) * 
	(total_gain_pct/100) * 
	(liquidity_pool_composition/100+(final_price/initial_price)*(100-liquidity_pool_composition)/100)
)

costs = (
	deposited_amount *
	leverage *
	(total_interest_pct/100)
)

profits = returns - costs

profit_pct = profits / deposited_amount * 100

# st.markdown("##### Initial Amounts:")
with st.expander("Initial Amounts:"):
	col1, col2 = st.columns(2)
	col1.metric("initial asset amount", f'{initial_asset_amount:.4f}')
	col2.metric("initial liability amount", f'{initial_liability_amount:.4f}')

# st.markdown("##### After converting to secondary token (to add it to the pool):")
with st.expander("After converting to secondary token (to add it to the pool):"):
	col1, col2 = st.columns(2)
	col1.metric("initial deposited token pool amount", f'{initial_deposited_token_pool_amount:.4f}')
	col2.metric("initial secondary token pool amount", f'{initial_secondary_token_pool_amount:.4f}')

# st.markdown("##### After staking returns, rewards and secondary token price change:")
with st.expander("After staking returns, rewards and secondary token price change:"):
	col1, col2 = st.columns(2)
	col1.metric("final deposited token pool amount", f'{final_deposited_token_pool_amount:.4f}')
	col2.metric("final secondary token pool amount", f'{final_secondary_token_pool_amount:.4f}')

# st.markdown("##### Final Amounts:")
with st.expander("Final Amounts:"):
	col1, col2 = st.columns(2)
	col1.metric("final asset amount", f'{final_asset_amount:.4f}')
	col2.metric("final liability amount (same as initial)", f'{final_liability_amount:.4f}')

# st.markdown("##### Results:")
with st.expander("Results:"):
	col1, col2 = st.columns(2)
	col1.metric("returns", f'{returns:.4f}')
	col2.metric("costs", f'{costs:.4f}')

# st.markdown("##### Profits (before tx costs):")
with st.expander("Profits (before tx costs):"):
	col1, col2 = st.columns(2)
	col1.metric("profits", f'{profits:.4f}')
	col2.metric("profits (% of the deposited amount)", f'{profit_pct:.4f}')

st.subheader("Tx Cost Results:")

# st.markdown("##### Estimated tx costs:")
with st.expander("Estimated tx costs:"):
	col1, col2 = st.columns(2)
	col1.metric(f'estimated tx cost to enter the strategy:', f'{gas_cost_entry} ETH')
	col2.metric(f'estimated tx cost to exit the strategy:', f'{gas_cost_exit} ETH')

# st.markdown("##### Profits (after tx costs):")
with st.expander("Profits (after tx costs):"):
	col1, col2 = st.columns(2)
	col1.metric("profits (after tx costs)", f'{profits - gas_cost_entry - gas_cost_exit:.4f}')
	col2.metric("profits (% of the deposited amount)", f'{(profits - gas_cost_entry - gas_cost_exit) / deposited_amount * 100:.4f}')

# st.markdown("##### Time to breakeven (in days):")
with st.expander("Time to breakeven (in days):"):
	def profit(days_invested, estimated_apy, borrowing_interest_rate, deposited_amount, leverage, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit):
		# apy (yield compounds)
		daily_gain = ((1+estimated_apy/100)**(1/365)-1)
		total_gain_pct =  ((1+daily_gain)**days_invested-1)*100
		# apr (interest doesn't compound)
		# daily_borrowing_interest_rate = ((1+borrowing_interest_rate/100)**(1/365)-1)
		# total_interest_pct =  ((1+daily_borrowing_interest_rate)**days_invested-1)*100
		daily_borrowing_interest_rate = (borrowing_interest_rate/100)/365
		total_interest_pct =  (daily_borrowing_interest_rate*days_invested)*100

		initial_asset_amount = deposited_amount * (1 + leverage)
		initial_liability_amount = (deposited_amount * leverage) * (1 + total_interest_pct / 100)

		initial_deposited_token_pool_amount = initial_asset_amount * liquidity_pool_composition / 100
		initial_secondary_token_pool_amount = (initial_asset_amount * (100 - liquidity_pool_composition) / 100)/initial_price

		final_deposited_token_pool_amount = initial_deposited_token_pool_amount * (1+total_gain_pct/100)
		final_secondary_token_pool_amount = initial_secondary_token_pool_amount * (1+total_gain_pct/100)

		final_asset_amount = final_deposited_token_pool_amount + (final_secondary_token_pool_amount * final_price)
		final_liability_amount = initial_liability_amount

		returns = (
			deposited_amount * 
			(leverage  + 1) * 
			(total_gain_pct/100) * 
			(liquidity_pool_composition/100+(final_price/initial_price)*(100-liquidity_pool_composition)/100)
		)

		costs = (
			deposited_amount *
			leverage *
			(total_interest_pct/100)
		)

		profits = returns - costs - gas_cost_entry - gas_cost_exit
		# profits = (1+daily_gain)**days_invested
		# profits = returns
		return profits
	
	try:
		profit_per_day = lambda x: profit(x, estimated_apy, borrowing_interest_rate, deposited_amount, leverage, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit)
		days_to_breakeven = optimize.brenth(lambda x: profit_per_day(x), 0, 10_000)
		st.metric("days to breakeven", f'{days_to_breakeven:.4f}')
	except Exception as e:
		st.metric("days to breakeven", f'never')

st.subheader("Visualizing the impact of the parameters:")

# # st.markdown("##### Profits Over time:")
# with st.expander("Profits Over time:"):
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=np.arange(0, 365, 1), y=[profit(x)/deposited_amount for x in np.arange(0, 365, 1)], name="profit"))
# 	fig.update_layout(
# 		title="Profit over time",
# 		xaxis_title="days invested",
# 		yaxis_title="profit (ETH)"
# 	)
# 	st.plotly_chart(fig, )

# st.markdown("##### APY vs final profit:")
with st.expander("APY vs final profit:"):
	profit_per_apr = lambda x: profit(days_invested, x, borrowing_interest_rate, deposited_amount, leverage, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit)/deposited_amount

	# try:
	# 	days_to_breakeven = optimize.brenth(lambda x: profit_per_apr(x), 0, 10_000)
	# 	st.metric(f"APY to breakeven after {days_invested} days", f'{days_to_breakeven:.4f}')
	# except Exception as e:
	# 	st.metric("days to breakeven", f'never')
	col1, col2, col3 = st.columns(3)
	min_apy = col1.number_input("min APY to simulate", 0.0, 250_000.0, 0.0)
	max_apy = col2.number_input("max APY to simulate", 0.0, 250_000.0, 12.5)
	step_size = col3.number_input("step size", 0.0, 250_000.0, 0.01, key="apy_step_size")

		
	fig = go.Figure()
	vec = np.arange(min_apy, max_apy, step_size)
	fig.add_trace(go.Scatter(x=vec, y=[profit_per_apr(x) for x in vec], name="profit"))
	fig.update_layout(
		title=f"Profit after {days_invested} days vs APY",
		xaxis_title="APY (%)",
		yaxis_title="profit (%)"
	)
	st.plotly_chart(fig, )

with st.expander("Leverage vs final profit:"):
	profit_per_leverage = lambda x: profit(days_invested, estimated_apy, borrowing_interest_rate, deposited_amount, x, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit)/deposited_amount

	# try:
	# 	days_to_breakeven = optimize.brenth(lambda x: profit_per_leverage(x), 0, 10_000)
	# 	st.metric(f"APY to breakeven after {days_invested} days", f'{days_to_breakeven:.4f}')
	# except Exception as e:
	# 	st.metric("days to breakeven", f'never')
	col1, col2, col3 = st.columns(3)
	min_leverage = col1.number_input("min leverage to simulate", 0.0, 250_000.0, 0.0)
	max_leverage = col2.number_input("max leverage to simulate", 0.0, 250_000.0, 10.0)
	step_size = col3.number_input("step size", 0.0, 250_000.0, 0.10, key="leverage_step_size")

		
	fig = go.Figure()
	vec = np.arange(min_leverage, max_leverage, step_size)
	fig.add_trace(go.Scatter(x=vec, y=[profit_per_leverage(x) for x in vec], name="profit"))
	fig.update_layout(
		title=f"Profit after {days_invested} days vs leverage",
		xaxis_title="leverage",
		yaxis_title="profit (%)"
	)
	st.plotly_chart(fig, )

with st.expander("Leverage and apy vs final profit:"):

	col1, col2, col3 = st.columns(3)
	min_apy = col1.number_input("min APY to simulate", 0.0, 250_000.0, 0.0, key="apy_min_apy_2")
	max_apy = col2.number_input("max APY to simulate", 0.0, 250_000.0, 25.0, key="apy_max_apy_2")
	step_size = col3.number_input("step size", 0.0, 250_000.0, 0.10, key="apy_step_size_2")
	min_leverage = col1.number_input("min leverage to simulate", 0.0, 250_000.0, 0.0, key="leverage_min_leverage_2")
	max_leverage = col2.number_input("max leverage to simulate", 0.0, 250_000.0, 10.0, key="leverage_max_leverage_2")
	step_size = col3.number_input("step size", 0.0, 250_000.0, 0.10, key="leverage_step_size_2")

	# plot surface
	fig = go.Figure()
	vec_apy = np.arange(min_apy, max_apy, step_size)
	vec_leverage = np.arange(min_leverage, max_leverage, step_size)
	fig.add_trace(go.Surface(x=vec_apy, y=vec_leverage, z=[[profit(days_invested, x, borrowing_interest_rate, deposited_amount, y, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit)/deposited_amount for x in vec_apy] for y in vec_leverage]))
	fig.update_layout(
		title=f"Profit surface after {days_invested} days vs APY vs leverage",
		scene = dict(
			xaxis_title="APY (%)",
			yaxis_title="leverage",
			zaxis_title="profit (%)"
		),
		height=800,
	)
	# add surface opacity
	fig.update_traces(opacity=0.75)

	fig.update_traces(contours_z=dict(show=True, usecolormap=True,project_z=True, highlightcolor='black'))

	st.plotly_chart(fig, )

	# plot contour
	fig = go.Figure()
	vec_apy = np.arange(min_apy, max_apy, step_size)
	vec_leverage = np.arange(min_leverage, max_leverage, step_size)
	fig.add_trace(
		go.Contour(
			x=vec_apy, 
			y=vec_leverage, 
			z=[[profit(days_invested, x, borrowing_interest_rate, deposited_amount, y, liquidity_pool_composition, initial_price, gas_cost_entry, gas_cost_exit)/deposited_amount for x in vec_apy] for y in vec_leverage],
			colorscale='RdBu',
			contours_coloring=None, # can also be 'lines', or 'none' or 'heatmap'
			line_smoothing=0,
		),
	)
	fig.update_layout(
		title=f"Profit countours after {days_invested} days vs APY vs leverage",
		scene = dict(
			xaxis_title="APY (%)",
			yaxis_title="leverage",
			zaxis_title="profit (%)"
		),
		# height=800,
	)
	# set title
	fig.update_xaxes(title_text="APY (%)")
	fig.update_yaxes(title_text="leverage")

	# showlabels=True

	st.plotly_chart(fig, )
