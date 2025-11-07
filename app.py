import streamlit as st
from vector_store import build_faiss
from qa_engine import load_qa, ask_question
from prompts import make_prompt

def main():
    st.title("📑 Advanced Policy Q/A Bot (Groq + LangChain)")

    uploaded = st.file_uploader("Upload policy PDF", type=["pdf"])
    if uploaded:
        with open("uploaded_policy.pdf", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Saved uploaded_policy.pdf — building index")
        build_faiss("uploaded_policy.pdf")
        st.success("Index built — now ask a question")

    role = st.selectbox("Role", ["employee","client"])
    length = st.selectbox("Answer length", ["short","medium","long"])
    q = st.text_input("Ask a question about the policy")

    if st.button("Ask"):
        qa = load_qa()
        system_prompt = make_prompt(role, length)
        answer, docs = ask_question(qa, q, system_prompt, q)

        st.markdown("**Answer**")
        st.write(answer)

        st.markdown("**Sources**")
        for d in docs:
            st.write(f"Page: {d.metadata.get('page')}")
            st.write(d.page_content[:400])

if __name__ == "__main__":
    main()
