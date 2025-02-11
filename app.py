import streamlit as st
from openai import OpenAI
import pandas as pd
import altair as alt
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np

# Set the page configuration to use the full screen width.
st.set_page_config(
    page_title="Personal AI Accountant",
    page_icon="💰",
    layout="wide"
)

# Replace OpenAI setup with Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# --- Database Setup: Unified Transactions Table ---
engine = create_engine("sqlite:///budget_tracker.db", connect_args={"check_same_thread": False})
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    date = Column(String, nullable=False)         # Stored as YYYY-MM-DD
    type = Column(String, nullable=False)           # "Income", "Expense", or "Investment"
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String)                    # For investments, we store the growth percentage as a string

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# ----------------------- Automated Transaction Classification -----------------------
def auto_classify_transaction(description):
    """
    Uses AI to classify an expense description.
    Returns a string suggestion (e.g., "Fixed Expense (Need)" or "Variable Expense (Want)").
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a financial transaction classifier. Analyze the following expense description and suggest an appropriate category along with whether it is a 'Need' or 'Want'."},
                {"role": "user", "content": f"Description: {description}"}
            ],
            max_tokens=1024,
            temperature=0.5,
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        return f"Error: {e}"

# ----------------------- Global Insights Graph -----------------------
def global_insights():
    # Query all Income and Expense transactions.
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense"])).all()
    if not transactions:
        now = datetime.today()
        dummy = {"month": [now.strftime("%Y-%m")], "Income": [0], "Expense": [0], "Net Savings": [0]}
        df_monthly = pd.DataFrame(dummy)
    else:
        data = [{"date": trans.date, "type": trans.type, "amount": trans.amount} for trans in transactions]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)
        income_df = df[df["type"]=="Income"].groupby("month")["amount"].sum().reset_index()
        expense_df = df[df["type"]=="Expense"].groupby("month")["amount"].sum().reset_index()
        monthly = pd.merge(income_df, expense_df, on="month", how="outer", suffixes=("_income", "_expense")).fillna(0)
        monthly = monthly.sort_values("month")
        monthly["Net Savings"] = monthly["amount_income"] - monthly["amount_expense"]
        monthly = monthly.rename(columns={"amount_income": "Income", "amount_expense": "Expense"})
        df_monthly = monthly[["month", "Income", "Expense", "Net Savings"]]
    
    df_monthly["Wealth Growth (%)"] = df_monthly["Net Savings"].pct_change().fillna(0) * 100

    df_melt = df_monthly.melt(id_vars="month", value_vars=["Income", "Expense", "Net Savings"],
                               var_name="Metric", value_name="Value")
    chart_A = alt.Chart(df_melt).mark_line(point=True).encode(
        x=alt.X("month:N", title="Month"),
        y=alt.Y("Value:Q", title="Amount ($)"),
        color=alt.Color("Metric:N", legend=alt.Legend(title="Metric"))
    ).properties(width=700, height=300, title="Monthly Income, Expenses & Net Savings")
    chart_B = alt.Chart(df_monthly).mark_bar().encode(
        x=alt.X("month:N", title="Month"),
        y=alt.Y("Wealth Growth (%):Q", title="Wealth Growth (%)", scale=alt.Scale(zero=True)),
        tooltip=["month", "Wealth Growth (%)"]
    ).properties(width=700, height=300, title="Monthly Wealth Growth (%)")
    final_chart = alt.vconcat(chart_A, chart_B).resolve_scale(y='independent')
    st.altair_chart(final_chart, use_container_width=True)

# ----------------------- Expenses Tab with AI-enhanced Classification -----------------------
def expenses_tab():
    st.header("Expenses")
    
    # Move Budgeting Tips to the top
    with st.expander("Budgeting Tips"):
        st.markdown(
            """
**Keep Your Budget Simple**

1. **Identify your income after tax:** Enter your monthly after‑tax income to know exactly what money is available.
2. **Track your fixed expenses:** List all fixed monthly expenses such as rent/mortgage, insurance, and car payments.
3. **Track your variable expenses:** Monitor expenses that fluctuate (groceries, gas, entertainment) using apps, spreadsheets, or notebooks.
4. **Differentiate needs vs wants:** Label your expenses to focus on essential needs over non‑essential wants.
5. **Allocate your income:** Distribute your income into fixed expenses, variable expenses, and savings/investments.
6. **Automate savings:** Set up automatic transfers to save consistently each payday.
7. **Regularly review your budget:** Check your budget monthly to make adjustments and stay on track.
8. **Use budgeting apps:** Leverage technology to simplify tracking and analysis.
9. **Keep it simple:** A simple budget is easier to maintain consistently.
            """
        )

    with st.form("expense_form_db", clear_on_submit=True):
        expense_date = st.date_input("Expense Date", value=datetime.today())
        expense_description = st.text_input("Description")
        expense_amount = st.number_input("Amount (INR)", min_value=0.0, step=0.01)
        expense_category = st.selectbox("Category", 
                                        ["Rent", "Groceries", "Utilities", "Transportation", 
                                         "Entertainment", "Dining Out", "Healthcare", "Other"])
        expense_nature = st.selectbox("Expense Nature", ["Need", "Want"])
        
        # Auto-classification button
        if expense_description:
            if st.button("Auto‑classify Expense"):
                suggestion = auto_classify_transaction(expense_description)
                st.info(f"AI Suggestion: {suggestion}")
        
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            full_description = f"[{expense_nature}] " + (expense_description if expense_description else "")
            new_trans = Transaction(
                date=expense_date.strftime("%Y-%m-%d"),
                type="Expense",
                category=expense_category,
                amount=expense_amount,
                description=full_description
            )
            db.add(new_trans)
            db.commit()
            st.success("Expense added!")
    
    st.subheader("All Expenses")
    expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
    if expenses:
        for trans in expenses:
            st.write(f"{trans.date} | {trans.category} | ₹{trans.amount:.2f} | {trans.description}")
    else:
        st.info("No expenses added yet.")

# ----------------------- Income Tab -----------------------
def income_tab():
    st.header("Income")
    
    with st.form("income_form", clear_on_submit=True):
        income_date = st.date_input("Income Date", value=datetime.today())
        income_description = st.text_input("Description")
        income_amount = st.number_input("Amount (INR)", min_value=0.0, step=0.01)
        
        submitted = st.form_submit_button("Add Income")
        if submitted:
            new_income = Transaction(
                date=income_date.strftime("%Y-%m-%d"),
                type="Income",
                category="Income",  # Assuming all entries are categorized as Income
                amount=income_amount,
                description=income_description
            )
            db.add(new_income)
            db.commit()
            st.success("Income added!")

    # Reset Income button
    if st.button("Reset Income"):
        # Logic to reset income data (e.g., clear the database or reset specific entries)
        # This is a placeholder; implement the actual reset logic as needed
        db.query(Transaction).filter(Transaction.type == "Income").delete()
        db.commit()
        st.success("All income data has been reset.")

    st.subheader("All Income Data")
    incomes = db.query(Transaction).filter(Transaction.type == "Income").order_by(Transaction.date).all()
    if incomes:
        for trans in incomes:
            st.write(f"{trans.date} | {trans.category} | ₹{trans.amount:.2f}")
    else:
        st.info("No incomes added yet.")

# ----------------------- Investments Tab -----------------------
def investments_tab():
    st.header("Investments")
    with st.form("investment_form_db", clear_on_submit=True):
        investment_date = st.date_input("Investment Date", value=datetime.today())
        investment_type = st.selectbox("Investment Type", ["Stocks", "Bonds", "Real Estate", "Crypto", "Other"])
        investment_amount = st.number_input("Amount Invested (INR)", min_value=0.0, step=0.01)
        annual_growth_rate = st.number_input("Annual Growth Percentage (%)", step=0.1, format="%.2f")
        submitted = st.form_submit_button("Add Investment")
        if submitted:
            new_trans = Transaction(
                date=investment_date.strftime("%Y-%m-%d"),
                type="Investment",
                category=investment_type,
                amount=investment_amount,
                description=str(annual_growth_rate)
            )
            db.add(new_trans)
            db.commit()
            st.success("Investment added!")
    
    st.subheader("All Investments")
    investments = db.query(Transaction).filter(Transaction.type == "Investment").order_by(Transaction.date).all()
    if investments:
        for trans in investments:
            try:
                annual_growth_rate = float(trans.description)
            except:
                annual_growth_rate = 0.0
            inv_date = datetime.strptime(trans.date, "%Y-%m-%d")
            today = datetime.today()
            years_elapsed = (today - inv_date).days / 365.25
            current_value = trans.amount * ((1 + annual_growth_rate / 100) ** years_elapsed)
            st.write(f"{trans.date} | {trans.category} | Invested: ₹{trans.amount:.2f} | Annual Growth: {annual_growth_rate:.2f}% | Years: {years_elapsed:.2f} | Current Value: ₹{current_value:.2f}")
    else:
        st.info("No investments added yet.")

# ----------------------- Budget Planner & 75-10-15 Rule with Scenario Simulation -----------------------
def monitor_budget(monthly_income, fixed_expenses, variable_expenses):
    recommended_spending = monthly_income * 0.75
    actual_spending = fixed_expenses + variable_expenses
    if actual_spending > recommended_spending:
        st.error("Alert: Your spending exceeds the recommended 75% of your income. Consider reducing discretionary expenses.")
    else:
        st.success("Your spending is within the recommended range.")

def budget_planner_tab():
    st.header("Budget Planner & 75-10-15 Rule")
    st.markdown(
        """
The 75-10-15 rule divides your after‑tax income into three parts:
- **Spending (75%)** on needs and wants
- **Saving (10%)** for your cushion fund
- **Investing (15%)** for long‑term wealth building

Follow these steps to plan your budget:
    """
    )
    st.markdown(
        """
1. **Enter your monthly after‑tax income.**  
2. **Input your total fixed expenses** (e.g., rent, insurance, car payments).  
3. **Input your total variable expenses** (e.g., groceries, gas, entertainment).  
4. **Review your allocations:**  
   - Spending should be around 75% of your income.  
   - Saving should be at least 10% of your income.  
   - Investing should be around 15% of your income.
        """
    )
    
    st.subheader("Interactive Budget Calculator")
    monthly_income = st.number_input("Monthly After‑Tax Income (INR)", min_value=0.0, step=0.01)
    fixed_expenses = st.number_input("Total Fixed Expenses (INR)", min_value=0.0, step=0.01)
    variable_expenses = st.number_input("Total Variable Expenses (INR)", min_value=0.0, step=0.01)
    
    if st.button("Calculate Budget Allocation"):
        recommended_spending = monthly_income * 0.75
        recommended_saving = monthly_income * 0.10
        recommended_investing = monthly_income * 0.15
        actual_spending = fixed_expenses + variable_expenses
        st.write("### Recommended Allocations")
        st.write(f"**Spending (75%):** INR {recommended_spending:,.2f}")
        st.write(f"**Saving (10%):** INR {recommended_saving:,.2f}")
        st.write(f"**Investing (15%):** INR {recommended_investing:,.2f}")
        st.write("### Your Expenses")
        st.write(f"**Total Spending (Fixed + Variable):** INR {actual_spending:,.2f}")
        
        # Call the budget monitoring function
        monitor_budget(monthly_income, fixed_expenses, variable_expenses)
    
    st.markdown("---")
    st.markdown(
        """
*By following these budgeting steps and the 75‑10‑15 rule, you can effectively manage your money, build an emergency cushion, and invest for a secure future.*
        """
    )

# ----------------------- Predictive Analytics for Future Cash Flow -----------------------
def forecast_cash_flow():
    st.header("Future Cash Flow Forecast")
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense"])).all()
    if not transactions:
        st.info("Not enough data to forecast.")
        return

    data = [{"date": trans.date, "type": trans.type, "amount": trans.amount} for trans in transactions]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    income_df = df[df["type"]=="Income"].groupby("month")["amount"].sum().reset_index()
    expense_df = df[df["type"]=="Expense"].groupby("month")["amount"].sum().reset_index()
    monthly = pd.merge(income_df, expense_df, on="month", how="outer", suffixes=("_income", "_expense")).fillna(0)
    monthly = monthly.sort_values("month")
    monthly["Net Savings"] = monthly["amount_income"] - monthly["amount_expense"]
    monthly["month_index"] = range(len(monthly))
    x = monthly["month_index"].values
    y = monthly["Net Savings"].values

    if len(x) < 2:
        st.info("Not enough data for forecasting.")
        return

    # Simple linear regression
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    forecast_index = np.array(range(len(x), len(x)+3))
    forecast = poly(forecast_index)
    
    st.write("### Forecasted Net Savings for the Next 3 Months:")
    for i, val in enumerate(forecast, 1):
        st.write(f"Month {i}: INR {val:,.2f}")

# ----------------------- Scenario Simulation and Goal Setting -----------------------
def trajectory_tab():
    st.header("Trajectory & Goal Setting")
    
    # User inputs
    current_savings = st.number_input("Current Savings (INR)", min_value=0.0, step=0.01)
    monthly_savings = st.number_input("Expected Monthly Savings (INR)", min_value=0.0, step=0.01)
    months = st.number_input("Forecast Period (Months)", min_value=1, value=12)

    if st.button("Simulate"):
        # Calculate future savings
        future_savings = [current_savings + (monthly_savings * month) for month in range(months + 1)]
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            "Month": range(months + 1),
            "Future Savings": future_savings
        })

        # Plotting the graph
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Month:Q", title="Months"),
            y=alt.Y("Future Savings:Q", title="Total Savings (INR)"),
            tooltip=["Month", "Future Savings"]
        ).properties(
            width=700,
            height=400,
            title="Projected Savings Over Time"
        )

        st.altair_chart(chart, use_container_width=True)

        # Display final savings after the forecast period
        st.write(f"After {months} months, your projected total savings will be: INR {future_savings[-1]:,.2f}")

# ----------------------- Financial KPIs Tab -----------------------
def financial_kpis_tab():
    st.header("Financial Health")
    
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense", "Investment"])).all()
    if not transactions:
        st.info("Please add some transactions first.")
        return
    
    df = pd.DataFrame([{
        "date": trans.date,
        "type": trans.type,
        "category": trans.category,
        "amount": trans.amount,
        "description": trans.description
    } for trans in transactions])
    
    total_income = df[df["type"] == "Income"]["amount"].sum()
    total_expense = df[df["type"] == "Expense"]["amount"].sum()
    total_invested = df[df["type"] == "Investment"]["amount"].sum()
    
    def compute_current(row):
        try:
            growth = float(row["description"])
        except:
            growth = 0
        return row["amount"] * (1 + growth/100)
    
    df_invest = df[df["type"]=="Investment"].copy()
    if not df_invest.empty:
        df_invest["current_value"] = df_invest.apply(compute_current, axis=1)
        total_current_value = df_invest["current_value"].sum()
    else:
        total_current_value = 0

    if st.button("Get Financial Insights"):
        prompt = (
            f"Here is my financial summary:\n"
            f"Total Income: ₹{total_income:.2f}\n"
            f"Total Expenses: ₹{total_expense:.2f}\n"
            f"Net Savings: ₹{total_income - total_expense:.2f}\n"
            f"Total Invested: ₹{total_invested:.2f}\n"
            f"Current Investment Value: ₹{total_current_value:.2f}\n"
            f"Please provide actionable insights and recommendations to improve my overall financial health."
        )
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a top-tier financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            insights = response.choices[0].message.content.strip()
            st.subheader("AI Financial Insights")
            st.write(insights)
        except Exception as e:
            st.error(f"Error generating AI advice: {e}")

    # ----------------------- Forecast Cash Flow Section -----------------------
    st.subheader("Future Cash Flow Forecast")
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense"])).all()
    if not transactions:
        st.info("Not enough data to forecast.")
        return

    data = [{"date": trans.date, "type": trans.type, "amount": trans.amount} for trans in transactions]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    income_df = df[df["type"]=="Income"].groupby("month")["amount"].sum().reset_index()
    expense_df = df[df["type"]=="Expense"].groupby("month")["amount"].sum().reset_index()
    monthly = pd.merge(income_df, expense_df, on="month", how="outer", suffixes=("_income", "_expense")).fillna(0)
    monthly = monthly.sort_values("month")
    monthly["Net Savings"] = monthly["amount_income"] - monthly["amount_expense"]
    monthly["month_index"] = range(len(monthly))
    x = monthly["month_index"].values
    y = monthly["Net Savings"].values

    if len(x) < 2:
        st.info("Not enough data for forecasting.")
        return

    # Simple linear regression
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    forecast_index = np.array(range(len(x), len(x)+3))
    forecast = poly(forecast_index)
    
    st.write("### Forecasted Net Savings for the Next 3 Months:")
    for i, val in enumerate(forecast, 1):
        st.write(f"Month {i}: ₹{val:,.2f}")

# ----------------------- Tax Assistant Tab -----------------------
def tax_assistant_tab():
    st.header("Tax Assistant")
    if st.button("Get Tax Advice"):
        expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
        if not expenses:
            st.info("Please add some expense transactions first.")
            return
        
        prompt = "You are a knowledgeable tax advisor. Based on the following expense transactions, provide tax-saving tips and identify potential deductions:\n"
        for trans in expenses:
            prompt += f"- {trans.date}: {trans.category} expense of ${trans.amount:.2f}\n"
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable tax advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,
                temperature=0.5,
            )
            advice = response.choices[0].message.content.strip()
            st.subheader("Tax Advice")
            st.write(advice)
        except Exception as e:
            st.error(f"Error generating tax advice: {e}")

# ----------------------- Chatbot Tab -----------------------
def one_on_one_advisor_tab():
    st.header("1-on-1 Advisor")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your financial advisor..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Get AI response
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor."},
                    *st.session_state.messages
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response.choices[0].message.content)
            
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------- Main App Layout -----------------------
st.title("Your Personal AI Accountant")

left_col, right_col = st.columns([1, 1])

with left_col:
    with st.container():
        global_insights()

with right_col:
    with st.container():
        tabs = st.tabs([
            "Expenses", 
            "Income", 
            "Investments", 
            "Budget Planner", 
            "Financial Health", 
            "Tax Assistant",
            "1-on-1 Advisor",  
            "Trajectory",  
        ])
        
        with tabs[0]:
            st.write("Track and categorize your daily expenses to understand your spending habits.")
            expenses_tab()
        
        with tabs[1]:
            st.write("Log your income sources to keep track of your earnings.")
            income_tab()
        
        with tabs[2]:
            st.write("Manage and monitor your investments to see how they grow over time.")
            investments_tab()
        
        with tabs[3]:
            st.write("Create and manage your budget to ensure you stay on track with your finances.")
            budget_planner_tab()
        
        with tabs[4]:
            st.write("Get insights into your financial situation and track key performance indicators.")
            financial_kpis_tab()
        
        with tabs[5]:
            st.write("Receive guidance on tax-saving strategies and deductions based on your expenses.")
            tax_assistant_tab()
        
        with tabs[6]:
            st.write("Engage in a personalized chat with your financial advisor for tailored advice.")
            one_on_one_advisor_tab()
        
        with tabs[7]:
            st.write("Simulate your savings growth over time based on your current financial habits.")
            trajectory_tab()

