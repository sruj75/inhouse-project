import streamlit as st
from openai import OpenAI
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
    description = Column(String)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# ----------------------- Expenses Tab -----------------------
def expenses_tab():
    st.header("Expenses")
    # Expense Entry Form
    with st.form("expense_form_db", clear_on_submit=True):
        expense_date = st.date_input("Expense Date", value=datetime.today())
        expense_description = st.text_input("Description")
        expense_amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
        expense_category = st.selectbox("Category", 
                                        ["Rent", "Groceries", "Utilities", "Transportation", 
                                         "Entertainment", "Dining Out", "Healthcare", "Other"])
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            new_trans = Transaction(
                date=expense_date.strftime("%Y-%m-%d"),
                type="Expense",
                category=expense_category,
                amount=expense_amount,
                description=expense_description
            )
            db.add(new_trans)
            db.commit()
            st.success("Expense added!")
    
    # Display all expense transactions
    st.subheader("All Expenses")
    expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
    if expenses:
        for trans in expenses:
            st.write(f"{trans.date} | {trans.category} | ${trans.amount:.2f} | {trans.description}")
    else:
        st.info("No expenses added yet.")

# ----------------------- Income Tab -----------------------
def income_tab():
    st.header("Income")
    # Income Entry Form
    with st.form("income_form_db", clear_on_submit=True):
        income_date = st.date_input("Income Date", value=datetime.today())
        income_source = st.text_input("Income Source")
        income_amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Add Income")
        if submitted:
            new_trans = Transaction(
                date=income_date.strftime("%Y-%m-%d"),
                type="Income",
                category=income_source,  # storing the income source in the category field
                amount=income_amount,
                description=""
            )
            db.add(new_trans)
            db.commit()
            st.success("Income added!")
    
    # Display all income transactions
    st.subheader("All Incomes")
    incomes = db.query(Transaction).filter(Transaction.type == "Income").order_by(Transaction.date).all()
    if incomes:
        for trans in incomes:
            st.write(f"{trans.date} | {trans.category} | ${trans.amount:.2f}")
    else:
        st.info("No incomes added yet.")

# ----------------------- Investments Tab -----------------------
def investments_tab():
    st.header("Investments")
    # Investment Entry Form
    with st.form("investment_form_db", clear_on_submit=True):
        investment_date = st.date_input("Investment Date", value=datetime.today())
        investment_type = st.selectbox("Investment Type", ["Stocks", "Bonds", "Real Estate", "Crypto", "Other"])
        investment_amount = st.number_input("Amount Invested ($)", min_value=0.0, step=0.01)
        current_value = st.number_input("Current Value ($)", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("Add Investment")
        if submitted:
            # For simplicity, we'll store the current value as a string in the description field.
            new_trans = Transaction(
                date=investment_date.strftime("%Y-%m-%d"),
                type="Investment",
                category=investment_type,
                amount=investment_amount,
                description=str(current_value)
            )
            db.add(new_trans)
            db.commit()
            st.success("Investment added!")
    
    # Display all investment transactions
    st.subheader("All Investments")
    investments = db.query(Transaction).filter(Transaction.type == "Investment").order_by(Transaction.date).all()
    if investments:
        for trans in investments:
            st.write(f"{trans.date} | {trans.category} | Invested: ${trans.amount:.2f} | Current Value: ${trans.description}")
    else:
        st.info("No investments added yet.")

# ----------------------- Financial KPIs Tab -----------------------
def financial_kpis_tab():
    st.header("Financial KPIs")
    # Gather transactions for Income, Expense, and Investment
    transactions = db.query(Transaction).filter(Transaction.type.in_(["Income", "Expense", "Investment"])).all()
    if not transactions:
        st.info("Please add some transactions first.")
        return
    
    # Convert transactions to a DataFrame for aggregation.
    data = [{
        "date": trans.date,
        "type": trans.type,
        "category": trans.category,
        "amount": trans.amount,
        "description": trans.description
    } for trans in transactions]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    
    # Calculate KPIs
    total_income = df[df["type"] == "Income"]["amount"].sum()
    total_expense = df[df["type"] == "Expense"]["amount"].sum()
    total_invested = df[df["type"] == "Investment"]["amount"].sum()
    # For investments, convert stored current value (in description) to float if possible.
    try:
        total_current_value = sum(float(x) for x in df[df["type"] == "Investment"]["description"].tolist() if x.replace('.', '', 1).isdigit())
    except Exception:
        total_current_value = 0
    net_income = total_income - total_expense
    savings_rate = (net_income / total_income * 100) if total_income > 0 else 0
    investment_return = ((total_current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0

    st.subheader("Summary KPIs")
    st.write(f"**Total Income:** ${total_income:.2f}")
    st.write(f"**Total Expenses:** ${total_expense:.2f}")
    st.write(f"**Net Savings:** ${net_income:.2f}")
    st.write(f"**Savings Rate:** {savings_rate:.2f}%")
    st.write(f"**Total Invested:** ${total_invested:.2f}")
    st.write(f"**Current Investment Value:** ${total_current_value:.2f}")
    st.write(f"**Investment Return:** {investment_return:.2f}%")
    
    st.info("These KPIs are common metrics used by financial advisors and wealthy individuals to assess financial health.")
    
    if st.button("Get AI Financial Insights"):
        prompt = (
            f"Analyze the following financial summary and provide actionable insights, "
            f"including advice on optimizing savings, investments, and tax strategies. "
            f"Metrics: Total Income: ${total_income:.2f}, Total Expenses: ${total_expense:.2f}, "
            f"Net Savings: ${net_income:.2f}, Savings Rate: {savings_rate:.2f}%, "
            f"Total Invested: ${total_invested:.2f}, Current Investment Value: ${total_current_value:.2f}, "
            f"Investment Return: {investment_return:.2f}%."
        )
        try:
            response = client.chat.completions.create(
                model="deepseek-r1-distill-qwen-32b",
                messages=[
                    {"role": "system", "content": "You are a top-tier financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            insights = response.choices[0].message.content.strip()
            st.subheader("AI Financial Insights")
            st.write(insights)
        except Exception as e:
            st.error(f"Error generating AI advice: {e}")

# ----------------------- Tax Assistant Tab -----------------------
def tax_assistant_tab():
    st.header("Tax Assistant")
    # Retrieve only expense transactions.
    expenses = db.query(Transaction).filter(Transaction.type == "Expense").order_by(Transaction.date).all()
    if not expenses:
        st.info("Please add some expense transactions first.")
        return
    
    st.subheader("Your Expense Transactions")
    for trans in expenses:
        st.write(f"{trans.date} | {trans.category} | ${trans.amount:.2f} | {trans.description}")
    
    if st.button("Get Tax Advice"):
        prompt = "You are a knowledgeable tax advisor. Based on the following expense transactions, provide tax-saving tips and identify potential deductions:\n"
        for trans in expenses:
            prompt += f"- {trans.date}: {trans.category} expense of ${trans.amount:.2f}\n"
        try:
            response = client.chat.completions.create(
                model="deepseek-r1-distill-qwen-32b",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable tax advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.5,
            )
            advice = response.choices[0].message.content.strip()
            st.subheader("Tax Advice")
            st.write(advice)
        except Exception as e:
            st.error(f"Error generating tax advice: {e}")

# ----------------------- Main App Layout -----------------------
st.title("AI Personal Accountant")
tabs = st.tabs(["Expenses", "Income", "Investments", "Financial KPIs", "Tax Assistant"])

with tabs[0]:
    expenses_tab()
with tabs[1]:
    income_tab()
with tabs[2]:
    investments_tab()
with tabs[3]:
    financial_kpis_tab()
with tabs[4]:
    tax_assistant_tab()
