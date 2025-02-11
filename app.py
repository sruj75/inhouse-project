import streamlit as st
from openai import OpenAI
import pandas as pd
import altair as alt
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np

# Set full-screen layout and basic configuration.
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

# --- Database Setup ---
engine = create_engine("sqlite:///budget_tracker.db", connect_args={"check_same_thread": False})
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    date = Column(String, nullable=False)  # Format: YYYY-MM-DD
    type = Column(String, nullable=False)  # "Income", "Expense", or "Investment"
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String)  # For investments, stores growth percentage as a string

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# ----------------------- AI Functions with Concise Prompts -----------------------
def auto_classify_transaction(description):
    """Return a one-line classification: Category & 'Need' or 'Want'."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "You are a financial transaction classifier. "
                    "Provide a one-line, data-driven classification including the category and whether it is a 'Need' or 'Want'."
                )},
                {"role": "user", "content": f"Expense description: {description}"}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def get_financial_insights(total_income, total_expense, total_invested, total_current_value):
    prompt = (
        f"Financial Summary:\n"
        f"- Total Income: ₹{total_income:.2f}\n"
        f"- Total Expenses: ₹{total_expense:.2f}\n"
        f"- Net Savings: ₹{total_income - total_expense:.2f}\n"
        f"- Total Invested: ₹{total_invested:.2f}\n"
        f"- Current Investment Value: ₹{total_current_value:.2f}\n"
        "In 5 to 10 lines, provide concise, data-driven and actionable steps to improve my financial health."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a top-tier financial advisor. Answer in 5-10 concise lines with actionable steps."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating insights: {e}"

def get_tax_advice(expenses):
    prompt = "Based on these expense transactions, provide 5-10 concise, data-driven tax-saving tips and potential deductions:\n"
    for trans in expenses:
        prompt += f"- {trans.date}: {trans.category} expense of ₹{trans.amount:.2f}\n"
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a knowledgeable tax advisor. Keep your response to 5-10 short lines with actionable advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating tax advice: {e}"

def get_chat_response(messages):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor. Answer in 5-10 concise, actionable lines."},
                *messages
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ----------------------- Global Insights Graph -----------------------
def global_insights():
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense"])).all()
    if not transactions:
        now = datetime.today()
        df_monthly = pd.DataFrame({"month": [now.strftime("%Y-%m")], "Income": [0], "Expense": [0], "Net Savings": [0]})
    else:
        data = [{"date": trans.date, "type": trans.type, "amount": trans.amount} for trans in transactions]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").astype(str)
        income_df = df[df["type"] == "Income"].groupby("month")["amount"].sum().reset_index()
        expense_df = df[df["type"] == "Expense"].groupby("month")["amount"].sum().reset_index()
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
        y=alt.Y("Value:Q", title="Amount (₹)"),
        color=alt.Color("Metric:N", legend=alt.Legend(title="Metric"))
    ).properties(width=700, height=300, title="Monthly Overview")
    chart_B = alt.Chart(df_monthly).mark_bar().encode(
        x=alt.X("month:N", title="Month"),
        y=alt.Y("Wealth Growth (%):Q", title="Wealth Growth (%)", scale=alt.Scale(zero=True)),
        tooltip=["month", "Wealth Growth (%)"]
    ).properties(width=700, height=300, title="Monthly Wealth Growth")
    st.altair_chart(alt.vconcat(chart_A, chart_B).resolve_scale(y='independent'), use_container_width=True)

# ----------------------- Tabs -----------------------
def expenses_tab():
    st.header("Expenses")
    with st.expander("Budgeting Tips", expanded=True):
        st.markdown(
            """
1. Track fixed and variable expenses.
2. Label each as "Need" or "Want".
3. Review your budget monthly.
            """
        )
    with st.form("expense_form", clear_on_submit=True):
        expense_date = st.date_input("Expense Date", value=datetime.today())
        expense_description = st.text_input("Description")
        expense_amount = st.number_input("Amount (₹)", min_value=0.0, step=0.01)
        expense_category = st.selectbox("Category", ["Rent", "Groceries", "Utilities", "Transportation", 
                                                       "Entertainment", "Dining Out", "Healthcare", "Other"])
        expense_nature = st.selectbox("Expense Nature", ["Need", "Want"])
        if expense_description and st.button("Auto‑classify Expense", key="auto_expense"):
            st.info(f"AI Suggestion: {auto_classify_transaction(expense_description)}")
        if st.form_submit_button("Add Expense"):
            full_description = f"[{expense_nature}] {expense_description}"
            db.add(Transaction(
                date=expense_date.strftime("%Y-%m-%d"),
                type="Expense",
                category=expense_category,
                amount=expense_amount,
                description=full_description
            ))
            db.commit()
            st.success("Expense added!")
    st.subheader("Expense Log")
    expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
    if expenses:
        for trans in expenses:
            st.write(f"{trans.date} | {trans.category} | ₹{trans.amount:.2f} | {trans.description}")
    else:
        st.info("No expenses recorded yet.")

def income_tab():
    st.header("Income")
    with st.form("income_form", clear_on_submit=True):
        income_date = st.date_input("Income Date", value=datetime.today())
        income_description = st.text_input("Description")
        income_amount = st.number_input("Amount (₹)", min_value=0.0, step=0.01)
        if st.form_submit_button("Add Income"):
            db.add(Transaction(
                date=income_date.strftime("%Y-%m-%d"),
                type="Income",
                category="Income",
                amount=income_amount,
                description=income_description
            ))
            db.commit()
            st.success("Income added!")
    if st.button("Reset Income"):
        db.query(Transaction).filter(Transaction.type == "Income").delete()
        db.commit()
        st.success("Income data reset.")
    st.subheader("Income Log")
    incomes = db.query(Transaction).filter(Transaction.type == "Income").order_by(Transaction.date).all()
    if incomes:
        for trans in incomes:
            st.write(f"{trans.date} | ₹{trans.amount:.2f}")
    else:
        st.info("No income recorded yet.")

def investments_tab():
    st.header("Investments")
    with st.form("investment_form", clear_on_submit=True):
        investment_date = st.date_input("Investment Date", value=datetime.today())
        investment_type = st.selectbox("Investment Type", ["Stocks", "Bonds", "Real Estate", "Crypto", "Other"])
        investment_amount = st.number_input("Amount Invested (₹)", min_value=0.0, step=0.01)
        annual_growth_rate = st.number_input("Annual Growth (%)", step=0.1, format="%.2f")
        if st.form_submit_button("Add Investment"):
            db.add(Transaction(
                date=investment_date.strftime("%Y-%m-%d"),
                type="Investment",
                category=investment_type,
                amount=investment_amount,
                description=str(annual_growth_rate)
            ))
            db.commit()
            st.success("Investment added!")
    st.subheader("Investment Log")
    investments = db.query(Transaction).filter(Transaction.type == "Investment").order_by(Transaction.date).all()
    if investments:
        for trans in investments:
            try:
                growth = float(trans.description)
            except:
                growth = 0.0
            inv_date = datetime.strptime(trans.date, "%Y-%m-%d")
            years = (datetime.today() - inv_date).days / 365.25
            current_value = trans.amount * ((1 + growth/100) ** years)
            st.write(f"{trans.date} | {trans.category} | Invested: ₹{trans.amount:.2f} | Growth: {growth:.2f}% | Value: ₹{current_value:.2f}")
    else:
        st.info("No investments recorded yet.")

def budget_planner_tab():
    st.header("Budget Planner (75-10-15 Rule)")
    st.markdown(
        """
Divide your after‑tax income into:
- **75% Spending**
- **10% Savings**
- **15% Investing**
        """
    )
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, step=0.01)
    fixed_expenses = st.number_input("Total Fixed Expenses (₹)", min_value=0.0, step=0.01)
    variable_expenses = st.number_input("Total Variable Expenses (₹)", min_value=0.0, step=0.01)
    if st.button("Calculate"):
        spending_rec = monthly_income * 0.75
        saving_rec = monthly_income * 0.10
        investing_rec = monthly_income * 0.15
        actual_spending = fixed_expenses + variable_expenses
        st.write(f"**Spending:** ₹{spending_rec:,.2f} | **Savings:** ₹{saving_rec:,.2f} | **Investing:** ₹{investing_rec:,.2f}")
        st.write(f"**Your Spending:** ₹{actual_spending:,.2f}")
        if actual_spending > spending_rec:
            st.error("Spending exceeds recommended limits. Consider cutting discretionary costs.")
        else:
            st.success("Your spending is within the recommended range.")

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
    
    # Helper function to safely convert values to float.
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0
    
    df_invest = df[df["type"]=="Investment"].copy()
    if not df_invest.empty:
        # Use safe_float to convert the description; non-numeric descriptions default to 0.
        df_invest["current_value"] = df_invest.apply(
            lambda row: row["amount"] * (1 + safe_float(row["description"]) / 100), axis=1
        )
        total_current_value = df_invest["current_value"].sum()
    else:
        total_current_value = 0

    if st.button("Get Financial Insights"):
        insights = get_financial_insights(total_income, total_expense, total_invested, total_current_value)
        st.subheader("AI Insights")
        st.write(insights)
    
    st.subheader("Cash Flow Forecast")
    trans_data = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense"])).all()
    if trans_data:
        data = [{"date": trans.date, "type": trans.type, "amount": trans.amount} for trans in trans_data]
        df_flow = pd.DataFrame(data)
        df_flow["date"] = pd.to_datetime(df_flow["date"])
        df_flow["month"] = df_flow["date"].dt.to_period("M").astype(str)
        inc = df_flow[df_flow["type"]=="Income"].groupby("month")["amount"].sum().reset_index()
        exp = df_flow[df_flow["type"]=="Expense"].groupby("month")["amount"].sum().reset_index()
        monthly = pd.merge(inc, exp, on="month", how="outer", suffixes=("_inc", "_exp")).fillna(0)
        monthly = monthly.sort_values("month")
        monthly["Net Savings"] = monthly["amount_inc"] - monthly["amount_exp"]
        monthly["idx"] = range(len(monthly))
        if len(monthly) > 1:
            coeffs = np.polyfit(monthly["idx"], monthly["Net Savings"], 1)
            poly = np.poly1d(coeffs)
            forecast_idx = np.array(range(len(monthly), len(monthly)+3))
            forecast = poly(forecast_idx)
            st.write("**Next 3 Months Forecast:**")
            for i, val in enumerate(forecast, 1):
                st.write(f"Month {i}: ₹{val:,.2f}")
        else:
            st.info("Not enough data for forecasting.")

def tax_assistant_tab():
    st.header("Tax Assistant")
    if st.button("Get Tax Advice"):
        expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
        if not expenses:
            st.info("Please add some expenses first.")
            return
        advice = get_tax_advice(expenses)
        st.subheader("Tax Tips")
        st.write(advice)

def one_on_one_advisor_tab():
    st.header("1-on-1 Advisor")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask your financial advisor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        ai_response = get_chat_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

def trajectory_tab():
    st.header("Savings Trajectory")
    current_savings = st.number_input("Current Savings (₹)", min_value=0.0, step=0.01)
    monthly_savings = st.number_input("Monthly Savings (₹)", min_value=0.0, step=0.01)
    months = st.number_input("Forecast Period (Months)", min_value=1, value=12)
    if st.button("Simulate"):
        future = [current_savings + monthly_savings * m for m in range(months + 1)]
        df_future = pd.DataFrame({"Month": range(months + 1), "Savings": future})
        chart = alt.Chart(df_future).mark_line(point=True).encode(
            x=alt.X("Month:Q", title="Months"),
            y=alt.Y("Savings:Q", title="Total Savings (₹)"),
            tooltip=["Month", "Savings"]
        ).properties(width=700, height=400, title="Projected Savings")
        st.altair_chart(chart, use_container_width=True)
        st.write(f"After {months} months, projected savings: ₹{future[-1]:,.2f}")

# ----------------------- Main App Layout -----------------------
st.title("Your Personal AI Accountant")

left_col, right_col = st.columns([1, 1])
with left_col:
    global_insights()
with right_col:
    tabs = st.tabs([
        "Expenses", "Income", "Investments", "Budget Planner",
        "Financial Health", "Tax Assistant", "1-on-1 Advisor", "Trajectory"
    ])
    with tabs[0]:
        expenses_tab()
    with tabs[1]:
        income_tab()
    with tabs[2]:
        investments_tab()
    with tabs[3]:
        budget_planner_tab()
    with tabs[4]:
        financial_kpis_tab()
    with tabs[5]:
        tax_assistant_tab()
    with tabs[6]:
        one_on_one_advisor_tab()
    with tabs[7]:
        trajectory_tab()
