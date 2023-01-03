import streamlit as st
from shroomdk import ShroomDK
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from scipy import optimize

# ! TODOs:
# factor in withdrawal costs (swap fees)
# fix colors of 3d plots
# idk

# set page to white theme
st.set_page_config(layout="wide", page_title="simulatooor", page_icon="🧠", initial_sidebar_state="collapsed")

# st.markdown(df### ['gas_used'].mean())
#st.write(df)
st.header("Strategy Simulator")

with st.expander("Yo! Here's a boring blob of text talking about the project:", expanded=True):
	st.markdown("""

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

	You can also use this app to simulate simpler strategies:
	- set the leverage to 0 to ignore the lending and borrowing part (e.g. liquidity pool farming with no leverage)
	- set the liquidity pool ratio to 10% to ignore the conversion to liquidity pool tokens (e.g. leveraged farming a single token)
	- set the APY to 0% to simulate only the borrowing costs

	The default parameters are a real world example using the Notional wstETH/ETH leveraged vaults strategy.

	If you liked it, follow me on Twitter [@0xDoing](https://twitter.com/0xDoing) -- and feel free to reach out!

	The code is messy af, but you can find it [here](https://github.com/0xrdt/defi_simulatooor)
	""")

st.subheader('Inputs:')
col1, col2 = st.columns(2)
deposited_amount = col1.number_input('Deposited Amount', value=125.0, step=0.0001, 
	help="E.g. 100 if you're going to deposit 100 ETH in the strategy")
leverage = col2.number_input('Leverage (total asset amount = 1 + leverage)', value=2.0, step=0.01, 
	help="E.g. 2 if you're going to borrow 2x the deposited amount and your total assets will be 3x the deposited amount")
initial_price = col1.number_input('Initial Price (secondary token / deposited token price)', value=0.985, step=0.0001,
	help="E.g. 0.985 if the wstETH price is 0.985 ETH, you deposited ETH and the strategy uses the wstETH/ETH liquidity pool")
final_price = col2.number_input('Estimated Final Price (secondary token / deposited token price)', value=0.970, step=0.0001,
	help="E.g. 0.985 if the wstETH price is 0.985 ETH, you deposited ETH and the strategy uses the wstETH/ETH liquidity pool")
estimated_apy = col1.number_input('Estimated average APY (%)', value=5.75, step=0.01,
	help='E.g. 5.75 if you expect to receive a time-weighted average yield of 5.75%')
borrowing_interest_rate = col2.number_input('Estimated average borrowing interest rate (%)', value=5.06, step=0.01,
	help="E.g. 5.06 if you expect to pay a time-weighted average interest rate of 5.06% on your borrowed assets. If there's a one-time borrow fee, add it in the advanced inputs section")
liquidity_pool_composition = col1.number_input('Liquidity Pool Ratio (use the % of the deposited token)', value=50.0, step=0.01,
	help='E.g. 50 if you expect to deposit 50% of the deposited token and 50% of the secondary token in the liquidity pool')
days_invested = col2.number_input('Days Invested', value=60, step=1,
	help='E.g. 60 if you want to see your profits/costs/etc after 60 days')

st.subheader("Optional Tx Cost Inputs:")
col1, col2 = st.columns(2)
gas_price = col1.number_input('gas price when depositing', value=16.0, step=0.01, min_value=1.0, max_value=1000.0,
	help='E.g. 16 if the gas price is 16 gwei')
eth_price_vs_deposit_token = col2.number_input('ETH price relative to the deposited token', value=1.0, step=0.000001,
	help='E.g. 1400 if the ETH price is 1400 USDC and your deposit token is USDC')
gas_used_entry =  col1.number_input('gas cost of entering the strategy', value=1_250_000, step=1,
	help='E.g. 1_250_000 if the gas cost is 1.25 million gas units (you can set it to 0 to ignore it)')
gas_used_exit =  col2.number_input('gas cost of exiting the strategy', value=1_250_000, step=1,
	help='E.g. 1_250_000 if the gas cost is 1.25 million gas units (you can set it to 0 to ignore it)')

st.subheader("Advanced Inputs:")
col1, col2 = st.columns(2)
deposit_to_secondary_token_conversion_fee = col1.number_input('LP Swap Fee (deposit token to secondary token conversion) (%)', value=0.04, step=0.001,
	help='E.g. 0.04 if you expect to pay a 0.04% fee when swapping the deposited token to the secondary token before entering the liquidity pool')
borrow_fee = col2.number_input('Borrow fee (%)', value=0.03, step=0.01,
	help='E.g. 0.03 if you expect to pay a 0.03% one-time fee when borrowing the deposited token')


base_parameters = {
	'Deposited Amount': deposited_amount,
	'Leverage': leverage,
	'Initial Price': initial_price,
	'Final Price': final_price,
	'Estimated APY (%)': estimated_apy,
	'Borrowing Interest Rate': borrowing_interest_rate,
	'Liquidity Pool Ratio': liquidity_pool_composition,
	'Days Invested': days_invested,
	'Gas Price': gas_price,
	'ETH Price vs Deposit Token': eth_price_vs_deposit_token,
	'Gas Used Entry': gas_used_entry,
	'Gas Used Exit': gas_used_exit,
	'Gas Cost Entry': gas_used_entry * gas_price * eth_price_vs_deposit_token,
	'Gas Cost Exit': gas_used_exit * gas_price * eth_price_vs_deposit_token,
	'LP Swap Fee': deposit_to_secondary_token_conversion_fee,
	'Borrow Fee': borrow_fee,
}

def simulate(parameters: dict):
	deposited_amount = parameters['Deposited Amount']
	leverage = parameters['Leverage']
	initial_price = parameters['Initial Price']
	final_price = parameters['Final Price']
	estimated_apy = parameters['Estimated APY (%)']
	borrowing_interest_rate = parameters['Borrowing Interest Rate']
	liquidity_pool_composition = parameters['Liquidity Pool Ratio']
	days_invested = parameters['Days Invested']
	gas_price = parameters['Gas Price']
	eth_price_vs_deposit_token = parameters['ETH Price vs Deposit Token'] # eg ETH/USDC = 1400
	gas_used_entry = parameters['Gas Used Entry']
	gas_used_exit = parameters['Gas Used Exit']
	gas_cost_entry = parameters['Gas Cost Entry']
	gas_cost_exit = parameters['Gas Cost Exit']
	deposit_to_secondary_token_conversion_fee = parameters['LP Swap Fee']
	borrow_fee = parameters['Borrow Fee']

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
	initial_secondary_token_pool_amount = (
		initial_asset_amount * 
		(100 - deposit_to_secondary_token_conversion_fee)/100  * 
		(100 - liquidity_pool_composition)/100
	)/initial_price

	final_deposited_token_pool_amount = initial_deposited_token_pool_amount * (1+total_gain_pct/100)
	final_secondary_token_pool_amount = initial_secondary_token_pool_amount * (1+total_gain_pct/100)

	final_asset_amount = (
		final_deposited_token_pool_amount 
		+ (final_secondary_token_pool_amount * final_price) * (100 - deposit_to_secondary_token_conversion_fee)/100
	)
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

	profits = returns - costs - initial_liability_amount*borrow_fee/100
	profit_pct = profits / deposited_amount * 100

	gas_cost_entry = gas_price * (gas_used_entry) / 1e9
	gas_cost_exit = gas_price * (gas_used_exit) / 1e9

	profits_after_gas = profits - (gas_cost_entry + gas_cost_exit)*eth_price_vs_deposit_token

	profit_pct_after_gas = profits_after_gas / deposited_amount * 100

	return {
		'Initial Asset Amount': initial_asset_amount,
		'Initial Liability Amount': initial_liability_amount,
		'Initial Deposited Token Pool Amount': initial_deposited_token_pool_amount,
		'Initial Secondary Token Pool Amount': initial_secondary_token_pool_amount,
		'Final Deposited Token Pool Amount': final_deposited_token_pool_amount,
		'Final Secondary Token Pool Amount': final_secondary_token_pool_amount,
		'Final Asset Amount': final_asset_amount,
		'Final Liability Amount': final_liability_amount,
		'Costs': costs,
		'Returns': returns,
		'Profits': profits,
		'Profit %': profit_pct,
		'Gas Cost Entry': gas_cost_entry,
		'Gas Cost Exit': gas_cost_exit,
		'Profits After Gas': profits_after_gas,
		'Profit % After Gas': profit_pct_after_gas,
	}

st.subheader("Results:")

results = simulate(base_parameters)

initial_asset_amount = results['Initial Asset Amount']
initial_liability_amount = results['Initial Liability Amount']
initial_deposited_token_pool_amount = results['Initial Deposited Token Pool Amount']
initial_secondary_token_pool_amount = results['Initial Secondary Token Pool Amount']
final_deposited_token_pool_amount = results['Final Deposited Token Pool Amount']
final_secondary_token_pool_amount = results['Final Secondary Token Pool Amount']
final_asset_amount = results['Final Asset Amount']
final_liability_amount = results['Final Liability Amount']
costs = results['Costs']
returns = results['Returns']
profits = results['Profits']
profit_pct = results['Profit %']
gas_cost_entry = results['Gas Cost Entry']
gas_cost_exit = results['Gas Cost Exit']
profits_after_gas = results['Profits After Gas']
profit_pct_after_gas = results['Profit % After Gas']

if not st.checkbox("Show compact results"):
		
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

else:

	st.write(results)

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
	
	try:
		
		def profit_per_day(day):
			parameters = base_parameters.copy()
			parameters['Days Invested']=day
			return simulate(parameters)['Profits After Gas']

		days_to_breakeven = optimize.brenth(lambda x: profit_per_day(x), 0, 10_000)
		st.metric("days to breakeven", f'{days_to_breakeven:.4f}')
	except Exception as e:
		st.metric("days to breakeven", f'never')

st.subheader("Visualizing the impact of the parameters:")

with st.expander("1 output vs 1 parameter:"):

	st.warning(f"all other parameters will be set to the inputs you gave earlier")
	st.error(f"some plots may look linear, if that happens it's because the parameters are not high enough to show the exponential growth")

	col1, col2 = st.columns(2)
	choosen_parameter = col1.selectbox("Choose a parameter", list(base_parameters.keys()), index=list(base_parameters.keys()).index('Estimated APY (%)'))
	choosen_output = col2.selectbox("Choose an output", list(results.keys()), index=list(results.keys()).index('Profit % After Gas'))

	def output_per_parameter(parameter, choosen_parameter, choosen_output):
		parameters = base_parameters.copy()
		parameters[choosen_parameter]=parameter
		return simulate(parameters)[choosen_output]

	# try:
	# 	days_to_breakeven = optimize.brenth(lambda x: profit_per_apr(x), 0, 10_000)
	# 	st.metric(f"APY to breakeven after {days_invested} days", f'{days_to_breakeven:.4f}')
	# except Exception as e:
	# 	st.metric("days to breakeven", f'never')

	col1, col2, col3 = st.columns(3)
	min_param = col1.number_input(f"Min {choosen_parameter} to simulate", 0.0, 250_000.0, 0.0)
	max_param = col2.number_input(f"Max {choosen_parameter} to simulate", 0.0, 250_000.0, 12.5)
	step_size = col3.number_input("Step Size", 0.0, 250_000.0, 0.01, key="first_step_size")

		
	fig = go.Figure()
	vec = np.arange(min_param, max_param, step_size)
	fig.add_trace(go.Scatter(x=vec, y=[output_per_parameter(x, choosen_parameter, choosen_output) for x in vec], name="profit"))
	fig.update_layout(
		title=f"{choosen_output} vs {choosen_parameter}",
		xaxis_title=f"{choosen_parameter}",
		yaxis_title=f"{choosen_output}"
	)
	st.plotly_chart(fig, use_container_width=True)

with st.expander("1 output vs 2 parameters:"):

	st.warning(f"all other parameters will be set to the inputs you gave earlier")
	st.error(f"some plots may look linear, if that happens it's because the parameters are not high enough to show the exponential growth")

	col1, col2, col3 = st.columns(3)
	choosen_parameter_1 = col1.selectbox("Choose the first parameter", list(base_parameters.keys()), index=list(base_parameters.keys()).index('Estimated APY (%)'))
	choosen_parameter_2 = col2.selectbox("Choose the second parameter", list(base_parameters.keys()), index=list(base_parameters.keys()).index('Leverage'))
	choosen_output = col3.selectbox("Choose an output", list(results.keys()), index=list(results.keys()).index('Profit % After Gas'), key="second_output")

	min_param_1 = col1.number_input(f"Min {choosen_parameter_1} to simulate", 0.0, 250_000.0, 0.0, key="first_min_param")
	max_param_1 = col2.number_input(f"Max {choosen_parameter_1} to simulate", 0.0, 250_000.0, 12.5, key="first_max_param")
	step_size_1 = col3.number_input("Step Size", 0.0, 250_000.0, 0.1, key="second_step_size",)
	min_param_2 = col1.number_input(f"Min {choosen_parameter_2} to simulate", 0.0, 250_000.0, 0.0, key="second_min_param")
	max_param_2 = col2.number_input(f"Max {choosen_parameter_2} to simulate", 0.0, 250_000.0, 12.5, key="second_max_param")
	step_size_2 = col3.number_input("Step Size", 0.0, 250_000.0, 0.1, key="third_step_size")

	def output_per_parameters(parameter_1, parameter_2, choosen_parameter_1, choosen_parameter_2, choosen_output):
		parameters = base_parameters.copy()
		parameters[choosen_parameter_1]=parameter_1
		parameters[choosen_parameter_2]=parameter_2
		return simulate(parameters)[choosen_output]

	if st.checkbox("Show 3D plot", value=True):
		# plot surface
		fig = go.Figure()
		vec_param_1 = np.arange(min_param_1, max_param_1, step_size_1)
		vec_param_2 = np.arange(min_param_2, max_param_2, step_size_2)
		z = [[output_per_parameters(x, y, choosen_parameter_1, choosen_parameter_2, choosen_output) for x in vec_param_1] for y in vec_param_2]
		fig.add_trace(go.Surface(x=vec_param_1, y=vec_param_2, z=z))
		fig.update_layout(
			title=f"[surface plot] {choosen_output} after {days_invested} days vs {choosen_parameter_1} and {choosen_parameter_2}",
			scene = dict(
				xaxis_title=f"{choosen_parameter_1}",
				yaxis_title=f"{choosen_parameter_2}",
				zaxis_title=f"{choosen_output}",
			),
			height=600,
		)
		# add surface opacity
		fig.update_traces(opacity=0.75)

		fig.update_traces(contours_z=dict(show=True, usecolormap=True,project_z=True, highlightcolor='black'))

		st.plotly_chart(fig, use_container_width=True)

	if st.checkbox("Show contour plot", value=True):
		# plot contour
		fig = go.Figure()
		fig.add_trace(
			go.Contour(
				x=vec_param_1, 
				y=vec_param_2, 
				z=z,
				colorscale='RdBu',
				contours_coloring=None, # can also be 'lines', or 'none' or 'heatmap'
				line_smoothing=0,
			),
		)
		fig.update_layout(
			title=f"[countour plot] {choosen_output} after {days_invested} days vs {choosen_parameter_1} vs {choosen_parameter_2}",
			scene = dict(
				xaxis_title=f"{choosen_parameter_1}",
				yaxis_title=f"{choosen_parameter_2}",
				zaxis_title=f"{choosen_output}"
			),
			# height=800,
		)
		# set title
		fig.update_xaxes(title_text=f"{choosen_parameter_1}", nticks=20)
		fig.update_yaxes(title_text=f"{choosen_parameter_2}", nticks=20)

		# showlabels=True

		st.plotly_chart(fig, use_container_width=True)
