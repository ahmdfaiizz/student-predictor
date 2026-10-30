import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓")
st.title("🎓 Student At-Risk Predictor")
st.caption("Early-warning tool: estimates the chance a student will fail "
           "the course, using background info and the first-period grade.")

LABELS = {
    "failures": "Past class failures", "absences": "Absences",
    "G1": "First-period grade", "studytime": "Weekly study time",
    "age": "Age", "Medu": "Mother's education", "Fedu": "Father's education",
    "goout": "Going out with friends", "health": "Health status",
    "freetime": "Free time after school",
    "sex_F": "Sex", "sex_M": "Sex", "higher_yes": "Wants higher education",
    "higher_no": "Wants higher education", "internet_yes": "Internet at home",
    "internet_no": "Internet at home", "schoolsup_yes": "Extra school support",
    "schoolsup_no": "Extra school support", "famsup_yes": "Family support",
    "famsup_no": "Family support",
}

with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 15, 22, 17)
        sex = st.selectbox("Sex", ["F", "M"])
        G1 = st.slider("First-period grade (0-20)", 0, 20, 10)
        failures = st.number_input("Past class failures", 0, 3, 0)
        absences = st.number_input("Absences", 0, 100, 4)
        studytime = st.slider("Weekly study time (1=<2h, 4=>10h)", 1, 4, 2)
    with c2:
        Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        health = st.slider("Health status (1-5)", 1, 5, 3)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet at home", ["yes", "no"])
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
    submitted = st.form_submit_button("Predict risk")

if submitted:
    payload = dict(age=age, sex=sex, Medu=Medu, Fedu=Fedu, studytime=studytime,
                   failures=failures, absences=absences, goout=goout,
                   health=health, freetime=freetime, G1=G1, higher=higher,
                   internet=internet, schoolsup=schoolsup, famsup=famsup)
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()

    prob, level = res["at_risk_probability"], res["risk_level"]
    colors = {"low": "green", "medium": "orange", "high": "red"}
    st.subheader("Result")
    st.markdown(f"Risk level: :{colors[level]}[**{level.upper()}**]")
    st.metric("Probability of failing", f"{prob:.0%}")
    st.progress(prob)

    st.subheader("Main factors")
    for f in res["top_factors"]:
        name = LABELS.get(f["feature"], f["feature"])
        icon = "🔺" if f["direction"] == "raises risk" else "🔻"
        st.write(f"{icon} **{name}** {f['direction']}")